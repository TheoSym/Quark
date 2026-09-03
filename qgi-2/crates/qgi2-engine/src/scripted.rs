//! A scripted engine: runs the whole turn loop with no model and no server.
//!
//! The turn loop is the most consequential code in QGI-2 and, without this,
//! the least exercised: every model step goes over HTTP, so `Session::round`
//! could only be tested against a live deployment. [`ScriptedEngine`]
//! implements [`Engine`] with canned responses, so the loop — plan, tool
//! rounds, extract, verify, commit, answer, metrics — runs end to end in a
//! unit test, deterministically, in milliseconds.
//!
//! # How it decides what to return
//!
//! It does not need to be told which step is calling. Every structured step
//! carries its schema, and the schemas differ in their `required` fields:
//! `["steps","needs_tools"]` is the plan step, `["facts"]` is extract,
//! `["entry_points"]` is route, `["tool","arguments"]` is tool-args. A request
//! with no schema is the answer step. Dispatching on the schema rather than on
//! a side channel means the fake exercises the same request the real engine
//! would receive, including the constraint it is supposed to honour.
//!
//! # What it deliberately does not fake
//!
//! It does not constrain its output to the schema. A test that wants to see
//! the loop reject malformed model output can script exactly that — the
//! reason `run_structured` re-validates after guided decoding is that real
//! deployments sometimes ignore the constraint too.

use crate::endpoint::{Endpoint, EngineKind};
use crate::metrics::AcceptanceSnapshot;
use crate::types::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, PromptTokensDetails, Usage};
use crate::Engine;
use anyhow::Result;
use async_trait::async_trait;
use qgi2_spec_types::Speculation;
use serde_json::{Value, json};
use std::sync::Mutex;

/// What the scripted engine answers for each kind of step.
#[derive(Debug, Clone)]
pub struct Script {
    /// Plan responses, consumed in order. The last one repeats.
    pub plans: Vec<Value>,
    /// Extract responses, consumed in order. The last one repeats.
    pub extracts: Vec<Value>,
    /// Route responses, consumed in order. The last one repeats.
    pub routes: Vec<Value>,
    /// Tool-args responses, consumed in order. The last one repeats.
    pub tool_args: Vec<Value>,
    /// Answer texts, consumed in order. The last one repeats.
    pub answers: Vec<String>,
    /// Usage attached to every response. `cached_tokens` is what the cache
    /// metric reads, so tests can drive it above or below the threshold.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    /// Acceptance the engine reports from its metrics endpoint.
    pub accept_length: Option<f64>,
    /// Whether `embed` succeeds. `false` exercises the degraded-retrieval path.
    pub embedder_up: bool,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            plans: vec![json!({ "steps": [{ "intent": "answer directly" }], "needs_tools": false })],
            extracts: vec![json!({ "facts": [] })],
            routes: vec![json!({ "entry_points": [] })],
            tool_args: vec![json!({ "tool": "read", "arguments": { "path": "a.rs" } })],
            answers: vec!["done".to_string()],
            prompt_tokens: 1000,
            completion_tokens: 50,
            cached_tokens: 900,
            accept_length: Some(2.5),
            embedder_up: true,
        }
    }
}

impl Script {
    /// A plan that asks for one tool.
    pub fn plan_calling(tool: &str, intent: &str) -> Value {
        json!({ "steps": [{ "intent": intent, "tool": tool }], "needs_tools": true })
    }

    /// A plan that answers directly.
    pub fn plan_answering() -> Value {
        json!({ "steps": [{ "intent": "answer" }], "needs_tools": false })
    }

    /// An extraction of the given `(subject, relation, object, confidence)`
    /// triples.
    pub fn extracting(facts: &[(&str, &str, &str, f32)]) -> Value {
        let facts: Vec<Value> = facts
            .iter()
            .map(|(s, r, o, c)| {
                json!({ "subject": s, "relation": r, "object": o, "confidence": c })
            })
            .collect();
        json!({ "facts": facts })
    }
}

/// Every request the engine saw, for assertions about what the loop sent.
#[derive(Debug, Clone)]
pub struct Seen {
    pub model: String,
    pub step: SeenStep,
    pub system: String,
    pub user: String,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub streamed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeenStep {
    Plan,
    Extract,
    Route,
    ToolArgs,
    Answer,
    Unknown,
}

pub struct ScriptedEngine {
    kind: EngineKind,
    script: Script,
    cursors: Mutex<Cursors>,
    seen: Mutex<Vec<Seen>>,
    embed_calls: Mutex<Vec<Vec<String>>>,
}

#[derive(Default)]
struct Cursors {
    plan: usize,
    extract: usize,
    route: usize,
    tool_args: usize,
    answer: usize,
}

impl ScriptedEngine {
    pub fn new(script: Script) -> Self {
        Self {
            kind: EngineKind::Sglang,
            script,
            cursors: Mutex::new(Cursors::default()),
            seen: Mutex::new(Vec::new()),
            embed_calls: Mutex::new(Vec::new()),
        }
    }

    pub fn with_kind(mut self, kind: EngineKind) -> Self {
        self.kind = kind;
        self
    }

    /// Everything the loop sent, in order.
    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    /// Steps in the order they were called.
    pub fn steps_called(&self) -> Vec<SeenStep> {
        self.seen().into_iter().map(|s| s.step).collect()
    }

    /// Every batch of texts sent to `embed`.
    pub fn embed_calls(&self) -> Vec<Vec<String>> {
        self.embed_calls.lock().unwrap().clone()
    }

    fn classify(req: &ChatRequest) -> SeenStep {
        let Some(schema) = &req.schema else {
            return SeenStep::Answer;
        };
        let required: Vec<&str> = schema["required"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if required.contains(&"needs_tools") {
            SeenStep::Plan
        } else if required.contains(&"facts") {
            SeenStep::Extract
        } else if required.contains(&"entry_points") {
            SeenStep::Route
        } else if required.contains(&"arguments") {
            SeenStep::ToolArgs
        } else {
            SeenStep::Unknown
        }
    }

    /// Consume the next scripted response for a step; the last one repeats so
    /// a test scripting one extraction does not have to script one per round.
    fn next(list: &[Value], cursor: &mut usize) -> Value {
        let v = list
            .get(*cursor)
            .or_else(|| list.last())
            .cloned()
            .unwrap_or(Value::Null);
        *cursor += 1;
        v
    }

    fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.script.prompt_tokens,
            completion_tokens: self.script.completion_tokens,
            total_tokens: self.script.prompt_tokens + self.script.completion_tokens,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(self.script.cached_tokens),
            }),
            cached_tokens: None,
        }
    }
}

#[async_trait]
impl Engine for ScriptedEngine {
    fn kind(&self) -> EngineKind {
        self.kind
    }

    fn supports(&self, _: Speculation) -> bool {
        true
    }

    async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse> {
        let step = Self::classify(req);
        let (system, user) = {
            let sys = req
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let usr = req
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            (sys, usr)
        };
        self.seen.lock().unwrap().push(Seen {
            model: endpoint.model.clone(),
            step,
            system,
            user,
            temperature: req.sampling.temperature,
            max_tokens: req.sampling.max_tokens,
            streamed: req.stream,
        });

        let mut c = self.cursors.lock().unwrap();
        let text = match step {
            SeenStep::Plan => Self::next(&self.script.plans, &mut c.plan).to_string(),
            SeenStep::Extract => Self::next(&self.script.extracts, &mut c.extract).to_string(),
            SeenStep::Route => Self::next(&self.script.routes, &mut c.route).to_string(),
            SeenStep::ToolArgs => Self::next(&self.script.tool_args, &mut c.tool_args).to_string(),
            SeenStep::Answer => {
                let a = self
                    .script
                    .answers
                    .get(c.answer)
                    .or_else(|| self.script.answers.last())
                    .cloned()
                    .unwrap_or_default();
                c.answer += 1;
                a
            }
            SeenStep::Unknown => "{}".to_string(),
        };

        Ok(ChatResponse {
            id: "scripted".into(),
            model: endpoint.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::assistant(text),
                finish_reason: Some("stop".into()),
            }],
            usage: Some(self.usage()),
        })
    }

    async fn embed(&self, _: &Endpoint, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_calls.lock().unwrap().push(texts.to_vec());
        if !self.script.embedder_up {
            anyhow::bail!("scripted embedder is down");
        }
        // Deterministic and distinct per text: a 4-dim vector from the bytes.
        // Similar strings get similar vectors, which is enough for the
        // retrieval path to have something to rank.
        Ok(texts
            .iter()
            .map(|t| {
                let b = t.as_bytes();
                let mut v = [0f32; 4];
                for (i, byte) in b.iter().enumerate() {
                    v[i % 4] += *byte as f32;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0);
                v.iter().map(|x| x / norm).collect()
            })
            .collect())
    }

    async fn acceptance(&self, _: &Endpoint) -> Result<AcceptanceSnapshot> {
        Ok(match self.script.accept_length {
            Some(a) => AcceptanceSnapshot::Gauge { accept_length: a },
            None => AcceptanceSnapshot::Unavailable,
        })
    }

    async fn health(&self, _: &Endpoint) -> bool {
        true
    }

    fn launch_hint(&self, s: Speculation) -> String {
        format!("(scripted: {s})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> Endpoint {
        Endpoint::new("http://scripted/v1", "m", Speculation::Off)
    }

    #[tokio::test]
    async fn steps_are_recognised_from_their_schemas() {
        let e = ScriptedEngine::new(Script::default());
        let plan = ChatRequest::new(vec![ChatMessage::user("q")])
            .with_schema(json!({ "required": ["steps", "needs_tools"] }));
        let extract = ChatRequest::new(vec![ChatMessage::user("q")])
            .with_schema(json!({ "required": ["facts"] }));
        let answer = ChatRequest::new(vec![ChatMessage::user("q")]);
        e.chat(&endpoint(), &plan).await.unwrap();
        e.chat(&endpoint(), &extract).await.unwrap();
        e.chat(&endpoint(), &answer).await.unwrap();
        assert_eq!(
            e.steps_called(),
            vec![SeenStep::Plan, SeenStep::Extract, SeenStep::Answer]
        );
    }

    #[tokio::test]
    async fn the_last_scripted_response_repeats() {
        let e = ScriptedEngine::new(Script {
            answers: vec!["one".into(), "two".into()],
            ..Script::default()
        });
        let req = ChatRequest::new(vec![ChatMessage::user("q")]);
        let mut got = Vec::new();
        for _ in 0..3 {
            got.push(e.chat(&endpoint(), &req).await.unwrap().text().to_string());
        }
        assert_eq!(got, vec!["one", "two", "two"]);
    }

    #[tokio::test]
    async fn usage_carries_the_scripted_cache_figure() {
        let e = ScriptedEngine::new(Script {
            prompt_tokens: 200,
            cached_tokens: 50,
            ..Script::default()
        });
        let r = e
            .chat(&endpoint(), &ChatRequest::new(vec![ChatMessage::user("q")]))
            .await
            .unwrap();
        assert_eq!(r.usage.unwrap().cache_hit_rate(), Some(0.25));
    }

    #[tokio::test]
    async fn embeddings_are_deterministic_and_unit_length() {
        let e = ScriptedEngine::new(Script::default());
        let a = e.embed(&endpoint(), &["task:auth".into()]).await.unwrap();
        let b = e.embed(&endpoint(), &["task:auth".into()]).await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        assert_eq!(e.embed_calls().len(), 2);
    }

    #[tokio::test]
    async fn a_down_embedder_errors_rather_than_returning_garbage() {
        let e = ScriptedEngine::new(Script {
            embedder_up: false,
            ..Script::default()
        });
        assert!(e.embed(&endpoint(), &["x".into()]).await.is_err());
    }
}
