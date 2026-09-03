//! OpenAI-compatible routes.

use crate::model_name::{all_model_names, parse_model_name};
use crate::openai::{Message, ToolCallMessage, ToolDeclaration, read_transcript};
use crate::state::AppState;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use qgi2_turn::{DeferToCaller, NoTools, RoundInput, RoundOutcome, TurnResult};
use serde::Deserialize;
use serde_json::{Value, json};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/qgi2/metrics", get(metrics))
        .route("/qgi2/end", post(end_session))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<Message>,
    /// The caller's tool catalogue. QGI-2 masks it by mood, then constrains the
    /// argument decode to the chosen tool's own schema.
    #[serde(default)]
    pub tools: Vec<ToolDeclaration>,
    /// Accepted and ignored: QGI-2 chooses sampling per step from the persona's
    /// mood and profile. Honouring a client's temperature would silently break
    /// the Deterministic profile's reproducibility guarantee.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI's end-user identifier. Used as the session client id when no
    /// `X-QGI2-Session` header is sent.
    #[serde(default)]
    pub user: Option<String>,
}

/// Which client this request belongs to.
///
/// Header first, then the OpenAI `user` field, then none. jcode's named-provider
/// config can send a static header per instance (`headers = { X-QGI2-Session =
/// "..." }`), which is the intended way to keep two jcode clients on one persona
/// from serialising behind one session lock.
fn client_id(headers: &HeaderMap, user: Option<&str>) -> Option<String> {
    headers
        .get("x-qgi2-session")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| user.map(str::to_string))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let parsed = parse_model_name(&req.model);
    let client = client_id(&headers, req.user.as_deref());
    let key = crate::state::SessionStore::key(parsed.persona, client.as_deref());
    let transcript = read_transcript(&req.messages);

    if transcript.query.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(error_body("no user message in the request")),
        )
            .into_response();
    }

    let specs: Vec<_> = req.tools.iter().map(|t| t.to_spec()).collect();
    let has_tools = !specs.is_empty();
    let runner = DeferToCaller::new(specs);

    let input = RoundInput {
        query: transcript.query,
        tool_results: transcript.tool_results,
        round: transcript.round,
    };

    let session = state.store.get(parsed.persona, client.as_deref()).await;
    let outcome = {
        let mut session = session.lock().await;
        // A caller that sent no tools gets the no-tool runner, so the mood mask
        // admits nothing and the loop never plans a call it cannot emit.
        if has_tools {
            session.round(input, &runner).await
        } else {
            session.round(input, &NoTools).await
        }
    };

    // Crash insurance: after every turn, not just at shutdown. A server that
    // dies mid-session now loses at most the turn in flight.
    if let Err(e) = state.store.persist_session(&key).await {
        tracing::warn!(error = %e, %key, "could not persist session");
    }

    match outcome {
        Ok(RoundOutcome::CallTools { calls, result }) => {
            let tool_calls: Vec<ToolCallMessage> =
                calls.iter().map(ToolCallMessage::from_call).collect();
            let body = completion_body(
                &parsed.render(),
                &result,
                json!({
                    "role": "assistant",
                    "content": Value::Null,
                    "tool_calls": tool_calls,
                }),
                "tool_calls",
            );
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(RoundOutcome::Answered(result)) => {
            let body = completion_body(
                &parsed.render(),
                &result,
                json!({ "role": "assistant", "content": result.answer }),
                "stop",
            );
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(error_body(&format!("{e:#}"))),
        )
            .into_response(),
    }
}

fn completion_body(
    model: &str,
    result: &TurnResult,
    message: Value,
    finish_reason: &str,
) -> Value {
    let m = &result.metrics;
    json!({
        "id": format!("qgi2-{}", m.turn),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": m.planner_prompt_tokens + m.worker_prompt_tokens,
            "completion_tokens": m.planner_completion_tokens + m.worker_completion_tokens,
            "total_tokens": m.total_tokens(),
            // The real number from vLLM, so the caller's own cache UI reports
            // QGI-2's prefix-cache behaviour.
            "prompt_tokens_details": {
                "cached_tokens": m.planner_cached_tokens + m.worker_cached_tokens
            }
        },
        // Namespaced so a strict OpenAI client ignores it.
        "qgi2": {
            "facts_committed": result.committed.len(),
            "rejection_rate": m.rejection_rate,
            "segment_hashes": result.segment_hashes,
            "cache_outlook": result.cache_outlook,
            "mood_switched_to": result.mood_switched_to.map(|m| m.as_str()),
            "tool_rounds_exhausted": result.tool_rounds_exhausted,
            "retrieval_degraded": result.retrieval_degraded,
            "route_suggested_mood": result.route_suggested_mood,
            // Breaches ride along on every response: the spec calls a threshold
            // drop a bug, and a bug nobody is shown is a bug nobody fixes.
            "breaches": result.breaches,
        }
    })
}

/// End a session now rather than waiting for the idle sweep.
async fn end_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EndRequest>,
) -> impl IntoResponse {
    let persona = parse_model_name(req.model.as_deref().unwrap_or_default()).persona;
    let client = client_id(&headers, req.user.as_deref());
    let key = crate::state::SessionStore::key(persona, client.as_deref());
    match state.store.end(&key).await {
        Ok(ended) => (StatusCode::OK, Json(json!({ "ended": ended, "key": key }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body(&format!("{e:#}"))),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct EndRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

async fn models() -> impl IntoResponse {
    let data: Vec<Value> = all_model_names()
        .into_iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "qgi2" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    // Every live session, not just the default persona's: a metrics endpoint
    // that reported one session while five were running was misleading.
    let mut out = serde_json::Map::new();
    for key in state.store.live_keys().await {
        // `get` would create a session; read the live map directly instead.
        let Some(session) = state.store.peek(&key).await else { continue };
        let s = session.lock().await;
        out.insert(
            key,
            json!({
                "turns": s.metrics.turns.len(),
                "thresholds": s.metrics.thresholds,
                "latest_breaches": s.metrics.latest_breaches(),
                "facts": s.graph.len(),
                "embeddings": s.retrieval().embedding_count(),
            }),
        );
    }
    Json(json!({ "sessions": out, "idle_timeout_secs": state.store.idle_timeout().as_secs() }))
}

fn error_body(message: &str) -> Value {
    json!({ "error": { "message": message, "type": "qgi2_error" } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_turn::ToolCall;

    #[test]
    fn a_tool_declaration_becomes_a_spec_with_its_schema() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"qgi2/builder-traceable",
                "messages":[{"role":"user","content":"read it"}],
                "tools":[{"type":"function","function":{
                    "name":"read","description":"read a file",
                    "parameters":{"type":"object","required":["path"],
                                  "properties":{"path":{"type":"string"}}}}}]}"#,
        )
        .unwrap();
        assert_eq!(req.tools.len(), 1);
        let spec = req.tools[0].to_spec();
        assert_eq!(spec.name, "read");
        assert_eq!(spec.parameters["required"][0], "path");
    }

    #[test]
    fn a_request_without_tools_still_parses() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(req.tools.is_empty());
    }

    #[test]
    fn a_tool_call_response_uses_the_right_finish_reason() {
        let calls = [ToolCall {
            id: "c1".into(),
            tool: "read".into(),
            arguments: json!({"path": "a.rs"}),
        }];
        let msgs: Vec<ToolCallMessage> = calls.iter().map(ToolCallMessage::from_call).collect();
        let body = completion_body(
            "qgi2/builder-traceable",
            &TurnResult::default(),
            json!({"role":"assistant","content":Value::Null,"tool_calls":msgs}),
            "tool_calls",
        );
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["id"], "c1");
        assert!(body["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn an_answer_response_uses_stop() {
        let result = TurnResult {
            answer: "done".into(),
            ..TurnResult::default()
        };
        let body = completion_body(
            "qgi2/builder-traceable",
            &result,
            json!({"role":"assistant","content":"done"}),
            "stop",
        );
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["message"]["content"], "done");
        assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[test]
    fn usage_carries_cached_tokens_through_to_the_caller() {
        use qgi2_metrics::TurnMetrics;
        use qgi2_spec_types::ModelRole;
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 1000, 50, 900);
        let result = TurnResult {
            metrics: m,
            ..TurnResult::default()
        };
        let body = completion_body("m", &result, json!({}), "stop");
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 900);
    }

    #[test]
    fn a_client_temperature_is_accepted_and_ignored() {
        // Honouring it would silently break the Deterministic profile.
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"qgi2/builder-deterministic","messages":[{"role":"user","content":"x"}],"temperature":1.9}"#,
        )
        .unwrap();
        assert_eq!(req.temperature, Some(1.9));
        let persona = parse_model_name(&req.model).persona;
        assert!(
            qgi2_spec_types::Profile::Deterministic
                .apply_sampling(persona.mood.table().planner_sampling)
                .is_greedy()
        );
    }

    #[test]
    fn the_models_listing_is_every_persona() {
        assert!(all_model_names().contains(&"qgi2/builder-traceable".to_string()));
        assert!(all_model_names().contains(&"qgi2/companion-quick".to_string()));
    }
}
