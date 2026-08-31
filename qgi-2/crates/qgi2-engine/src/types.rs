//! Engine-agnostic request and response types.
//!
//! [`ChatRequest`] holds the *intent* — these messages, this sampling, under
//! this schema — and each backend renders it into its own wire body via
//! [`ChatRequest::to_openai_body`] plus whatever that engine calls a schema
//! constraint. The schema deliberately does **not** serialize itself: vLLM
//! wants `guided_json` and SGLang wants `response_format`, and a shared field
//! would have to pick one and be silently wrong on the other.

use anyhow::{Result, bail};
use qgi2_spec_types::Sampling;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// What the harness wants from one model call.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub sampling: Sampling,
    /// JSON schema for a structured step. Rendered per engine.
    pub schema: Option<Value>,
    /// Extra body fields a deployment needs.
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            sampling: Sampling::at_temperature(0.7),
            schema: None,
            extra: Map::new(),
        }
    }

    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// The fields both engines share, as an OpenAI chat body.
    ///
    /// Optional fields are omitted rather than sent as null: some
    /// OpenAI-compatible gateways reject explicit nulls.
    pub fn to_openai_body(&self, model: &str) -> Map<String, Value> {
        let mut body = Map::new();
        body.insert("model".into(), json!(model));
        body.insert("messages".into(), json!(self.messages));
        body.insert("temperature".into(), json!(self.sampling.temperature));
        body.insert("top_p".into(), json!(self.sampling.top_p));
        if let Some(seed) = self.sampling.seed {
            body.insert("seed".into(), json!(seed));
        }
        if let Some(max) = self.sampling.max_tokens {
            body.insert("max_tokens".into(), json!(max));
        }
        if self.sampling.batch_invariant {
            // Both engines expose batch-invariant kernels as a launch flag; the
            // request-level hint is passed through for deployments that read it
            // and is harmless where it is ignored.
            body.insert("batch_invariant".into(), json!(true));
        }
        for (k, v) in &self.extra {
            body.insert(k.clone(), v.clone());
        }
        body
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ChatResponse {
    pub fn text(&self) -> &str {
        self.choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    /// The OpenAI-standard location, used by current vLLM and SGLang.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Older SGLang builds report the count here instead. Read as a fallback so
    /// the cache metric does not silently read zero on those deployments —
    /// which, since the spec treats a cache drop as a bug, would send someone
    /// debugging a prefix that was fine all along.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

impl Usage {
    /// Tokens served from the prefix cache.
    pub fn cached_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .and_then(|d| d.cached_tokens)
            .or(self.cached_tokens)
            .unwrap_or(0)
    }

    /// Prefix-cache hit rate for this request.
    ///
    /// `None` rather than 0.0 when there were no prompt tokens: a zero-token
    /// request has no hit rate, and folding it in as a 0 would drag the session
    /// average down and make a healthy deployment look broken.
    pub fn cache_hit_rate(&self) -> Option<f64> {
        if self.prompt_tokens == 0 {
            return None;
        }
        Some(self.cached_tokens() as f64 / self.prompt_tokens as f64)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingDatum>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl EmbeddingResponse {
    /// Vectors in input order.
    ///
    /// Sorted by index because the API does not guarantee response order
    /// matches input order, and a mismatch would silently attach every
    /// embedding to the wrong node.
    pub fn into_ordered(self, expected: usize) -> Result<Vec<Vec<f32>>> {
        let mut data = self.data;
        data.sort_by_key(|d| d.index);
        if data.len() != expected {
            bail!(
                "embedder returned {} vectors for {expected} inputs",
                data.len()
            );
        }
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingDatum {
    #[serde(default)]
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::Profile;

    #[test]
    fn cache_hit_rate_reads_the_standard_location() {
        let u = Usage {
            prompt_tokens: 1000,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(900),
            }),
            ..Usage::default()
        };
        assert_eq!(u.cache_hit_rate(), Some(0.9));
    }

    #[test]
    fn an_older_sglang_shape_still_reports_cached_tokens() {
        // Reading zero here would send someone debugging a prefix that was fine.
        let u: Usage =
            serde_json::from_str(r#"{"prompt_tokens":1000,"cached_tokens":800}"#).unwrap();
        assert_eq!(u.cached_tokens(), 800);
        assert_eq!(u.cache_hit_rate(), Some(0.8));
    }

    #[test]
    fn the_standard_location_wins_when_both_are_present() {
        let u: Usage = serde_json::from_str(
            r#"{"prompt_tokens":1000,"cached_tokens":1,"prompt_tokens_details":{"cached_tokens":900}}"#,
        )
        .unwrap();
        assert_eq!(u.cached_tokens(), 900);
    }

    #[test]
    fn a_zero_token_request_has_no_hit_rate() {
        assert_eq!(Usage::default().cache_hit_rate(), None);
    }

    #[test]
    fn optional_fields_are_omitted_rather_than_null() {
        let body = ChatRequest::new(vec![ChatMessage::user("hi")]).to_openai_body("m");
        for key in ["seed", "max_tokens", "batch_invariant"] {
            assert!(body.get(key).is_none(), "{key} was serialized");
        }
    }

    #[test]
    fn deterministic_sampling_reaches_the_body() {
        let s = Profile::Deterministic.apply_sampling(qgi2_spec_types::Sampling::at_temperature(0.7));
        let body = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_sampling(s)
            .to_openai_body("m");
        assert_eq!(body["temperature"], json!(0.0));
        assert!(body.get("seed").is_some());
        assert_eq!(body["batch_invariant"], json!(true));
    }

    #[test]
    fn the_schema_does_not_serialize_itself() {
        // Each engine names it differently; a shared field would pick one and be
        // silently wrong on the other.
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_schema(json!({"type": "object"}));
        let body = req.to_openai_body("m");
        assert!(body.get("guided_json").is_none());
        assert!(body.get("response_format").is_none());
        assert!(req.schema.is_some());
    }

    #[test]
    fn embeddings_are_returned_in_input_order() {
        let r = EmbeddingResponse {
            data: vec![
                EmbeddingDatum {
                    index: 1,
                    embedding: vec![2.0],
                },
                EmbeddingDatum {
                    index: 0,
                    embedding: vec![1.0],
                },
            ],
            usage: None,
        };
        assert_eq!(r.into_ordered(2).unwrap(), vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn a_short_embedding_response_is_an_error_not_a_silent_misalignment() {
        let r = EmbeddingResponse {
            data: vec![EmbeddingDatum {
                index: 0,
                embedding: vec![1.0],
            }],
            usage: None,
        };
        assert!(r.into_ordered(2).is_err());
    }
}
