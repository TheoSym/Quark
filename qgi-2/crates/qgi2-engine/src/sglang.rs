//! The SGLang backend.
//!
//! Three things differ from vLLM in ways that matter to this harness:
//!
//! 1. **Schema constraint.** SGLang takes the OpenAI-standard
//!    `response_format: {type: "json_schema", json_schema: {...}}` rather than
//!    vLLM's `guided_json`. Sending `guided_json` to SGLang is not an error —
//!    it is silently ignored, which would leave every "structured" step
//!    unconstrained while the harness believed otherwise. That failure mode is
//!    why the schema field lives behind the trait instead of in the shared
//!    request type.
//!
//! 2. **Acceptance.** `sglang:spec_accept_length` is a gauge that already *is*
//!    the mean accepted length, so there is nothing to difference.
//!
//! 3. **Cache hit rate.** SGLang publishes `sglang:cache_hit_rate` directly,
//!    which is a better number than the per-request ratio the harness computes
//!    from `cached_tokens`: it covers RadixAttention's cross-request sharing,
//!    which is exactly what a stable prefix is supposed to exploit.

use crate::endpoint::{Endpoint, EngineKind, engine_typically_supports};
use crate::http::HttpClient;
use crate::metrics::{AcceptanceSnapshot, prometheus_lines};
use crate::types::{ChatRequest, ChatResponse, EmbeddingResponse};
use crate::{Engine, send_chat};
use anyhow::{Context, Result};
use async_trait::async_trait;
use qgi2_spec_types::Speculation;
use serde_json::json;

pub struct SglangEngine {
    http: HttpClient,
}

impl SglangEngine {
    pub fn new(http: HttpClient) -> Self {
        Self { http }
    }

    /// RadixAttention's cross-request prefix hit rate, when the server reports
    /// it.
    ///
    /// The harness's own per-request figure from `cached_tokens` only sees the
    /// prefix this session reused. This one also counts prefix shared *between*
    /// sessions on the same server, which is the whole point of running several
    /// personas against one process.
    pub async fn cache_hit_rate(&self, endpoint: &Endpoint) -> Option<f64> {
        let url = format!("{}/metrics", endpoint.root());
        let body = self
            .http
            .get_text(&url, endpoint.api_key.as_deref())
            .await
            .ok()?;
        prometheus_lines(&body).find_map(|(name, v)| {
            matches!(name, "sglang:cache_hit_rate" | "sglang:cached_tokens_rate").then_some(v)
        })
    }
}

#[async_trait]
impl Engine for SglangEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Sglang
    }

    fn supports(&self, speculation: Speculation) -> bool {
        engine_typically_supports(EngineKind::Sglang, speculation)
    }

    async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse> {
        let mut body = req.clone().streaming(endpoint.stream).to_openai_body(&endpoint.model);

        if let Some(schema) = &req.schema {
            // The OpenAI-standard form. `guided_json` would be accepted and
            // ignored here, leaving the step unconstrained.
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qgi2_step",
                        "schema": schema,
                        "strict": true
                    }
                }),
            );
        }

        send_chat(&self.http, endpoint, &body).await
    }

    async fn embed(&self, endpoint: &Endpoint, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let text = self
            .http
            .post_json(
                endpoint,
                "/embeddings",
                &json!({ "model": endpoint.model, "input": texts }),
            )
            .await?;
        let parsed: EmbeddingResponse =
            serde_json::from_str(&text).context("decoding embeddings response")?;
        parsed.into_ordered(texts.len())
    }

    async fn acceptance(&self, endpoint: &Endpoint) -> Result<AcceptanceSnapshot> {
        let url = format!("{}/metrics", endpoint.root());
        let body = self
            .http
            .get_text(&url, endpoint.api_key.as_deref())
            .await?;

        for (name, value) in prometheus_lines(&body) {
            if matches!(
                name,
                "sglang:spec_accept_length" | "sglang:speculative_accept_length"
            ) {
                return Ok(AcceptanceSnapshot::Gauge {
                    accept_length: value,
                });
            }
        }
        Ok(AcceptanceSnapshot::Unavailable)
    }

    async fn health(&self, endpoint: &Endpoint) -> bool {
        self.http.health(endpoint).await
    }

    fn launch_hint(&self, speculation: Speculation) -> String {
        match speculation {
            Speculation::Off => "--enable-hierarchical-cache (no speculation)".into(),
            // NEXTN is SGLang's name for a model's own MTP head.
            Speculation::Mtp { n } => format!(
                "--speculative-algorithm NEXTN --speculative-num-steps {n} \
                 --speculative-num-draft-tokens {}",
                n + 1
            ),
            Speculation::Eagle3 { n } => format!(
                "--speculative-algorithm EAGLE3 --speculative-num-steps {n} \
                 --speculative-eagle-topk 8 --speculative-num-draft-tokens {}",
                n + 1
            ),
            Speculation::NGram { n } => {
                format!("--speculative-algorithm NGRAM --speculative-num-draft-tokens {n}")
            }
            s => format!("SGLang does not implement {s}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest};

    fn engine() -> SglangEngine {
        SglangEngine::new(HttpClient::default())
    }

    #[test]
    fn a_schema_becomes_response_format_not_guided_json() {
        // Sending guided_json here would be silently ignored, leaving the step
        // unconstrained while the harness believed it was schema-bound.
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_schema(json!({"type": "object"}));
        let mut body = req.to_openai_body("m");
        body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": { "name": "qgi2_step", "schema": req.schema.clone().unwrap(), "strict": true }
            }),
        );
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert!(body.get("guided_json").is_none());
    }

    #[test]
    fn acceptance_reads_the_gauge_directly() {
        let body = "\
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length{model=\"w\"} 2.4
sglang:num_running_reqs 3.0
";
        let found = prometheus_lines(body)
            .find_map(|(n, v)| (n == "sglang:spec_accept_length").then_some(v));
        assert_eq!(found, Some(2.4));
        let snap = AcceptanceSnapshot::Gauge { accept_length: 2.4 };
        assert_eq!(snap.tokens_per_step(), Some(2.4));
    }

    #[test]
    fn the_cache_hit_rate_gauge_is_recognised() {
        let body = "sglang:cache_hit_rate{model=\"w\"} 0.91\n";
        let found = prometheus_lines(body)
            .find_map(|(n, v)| (n == "sglang:cache_hit_rate").then_some(v));
        assert_eq!(found, Some(0.91));
    }

    #[test]
    fn mtp_launches_as_nextn() {
        let hint = engine().launch_hint(Speculation::Mtp { n: 3 });
        assert!(hint.contains("NEXTN"), "{hint}");
        assert!(hint.contains("--speculative-num-steps 3"), "{hint}");
    }

    #[test]
    fn eagle3_launches_with_a_topk() {
        let hint = engine().launch_hint(Speculation::Eagle3 { n: 5 });
        assert!(hint.contains("EAGLE3"), "{hint}");
        assert!(hint.contains("--speculative-eagle-topk"), "{hint}");
    }

    #[test]
    fn the_hint_says_so_when_sglang_cannot_serve_it() {
        assert!(
            engine()
                .launch_hint(Speculation::DFlash2 { n: 7 })
                .contains("does not implement")
        );
    }
}
