//! Wire types for vLLM's OpenAI-compatible API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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

/// A chat completion request.
///
/// The guided-decoding fields are vLLM extensions rather than OpenAI standard,
/// which is why they are explicit fields here instead of being buried in an
/// `extra_body` blob: they are load-bearing for the spec's "every structured
/// step runs under a JSON schema" invariant, and a typo in a blob key would
/// silently disable constrained decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,

    /// vLLM guided decoding: constrain output to this JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_json: Option<Value>,
    /// Guided-decoding backend. Pinned by the Deterministic profile, where a
    /// backend change can alter which token the grammar admits first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_decoding_backend: Option<String>,

    /// Ask vLLM for usage on the final streaming chunk. Without this, a
    /// streamed request returns no `cached_tokens` and the cache-hit metric
    /// silently reads zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Anything else the deployment needs.
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty", default)]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: 0.7,
            top_p: 1.0,
            seed: None,
            max_tokens: None,
            stream: false,
            guided_json: None,
            guided_decoding_backend: None,
            stream_options: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Apply a step's sampling.
    pub fn with_sampling(mut self, s: &qgi2_spec_types::Sampling) -> Self {
        self.temperature = s.temperature;
        self.top_p = s.top_p;
        self.seed = s.seed;
        self.max_tokens = s.max_tokens;
        if s.batch_invariant {
            // vLLM exposes batch-invariant kernels through an engine flag; the
            // request-level hint is passed through for deployments that read
            // it, and is harmless where it is ignored.
            self.extra
                .insert("batch_invariant".into(), Value::Bool(true));
        }
        self
    }

    /// Constrain output to a schema.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.guided_json = Some(schema);
        self
    }

    /// Enable streaming and ask for the usage chunk.
    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl ChatResponse {
    /// The first choice's text, or an empty string.
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
    /// vLLM reports prefix-cache hits here. This is the number the spec's
    /// ">= 85% cache hit rate" metric is computed from.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    /// Tokens served from the prefix cache.
    pub fn cached_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0)
    }

    /// Prefix-cache hit rate for this request.
    ///
    /// Returns `None` rather than 0.0 when there were no prompt tokens: a
    /// zero-token request has no hit rate, and folding it in as a 0 would drag
    /// the session average down and make a healthy deployment look broken.
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingDatum {
    #[serde(default)]
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// One streamed chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::{Profile, Sampling};

    #[test]
    fn cache_hit_rate_reads_vllms_cached_tokens() {
        let u = Usage {
            prompt_tokens: 1000,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(900),
            }),
            ..Usage::default()
        };
        assert_eq!(u.cached_tokens(), 900);
        assert_eq!(u.cache_hit_rate(), Some(0.9));
    }

    #[test]
    fn a_zero_token_request_has_no_hit_rate_rather_than_zero() {
        // Folding it in as 0.0 would drag the session average down and make a
        // healthy deployment look like it breached the 85% threshold.
        let u = Usage::default();
        assert_eq!(u.cache_hit_rate(), None);
    }

    #[test]
    fn missing_cached_tokens_reads_as_zero_not_an_error() {
        let u = Usage {
            prompt_tokens: 100,
            ..Usage::default()
        };
        assert_eq!(u.cached_tokens(), 0);
        assert_eq!(u.cache_hit_rate(), Some(0.0));
    }

    #[test]
    fn usage_deserializes_from_vllms_shape() {
        let json = r#"{
            "prompt_tokens": 1200,
            "completion_tokens": 40,
            "total_tokens": 1240,
            "prompt_tokens_details": { "cached_tokens": 1024 }
        }"#;
        let u: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(u.cached_tokens(), 1024);
    }

    #[test]
    fn deterministic_sampling_reaches_the_wire() {
        let s = Profile::Deterministic.apply_sampling(Sampling::at_temperature(0.7));
        let req = ChatRequest::new("m", vec![ChatMessage::user("hi")]).with_sampling(&s);
        assert_eq!(req.temperature, 0.0);
        assert!(req.seed.is_some());
        assert_eq!(req.extra.get("batch_invariant"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn streaming_requests_ask_for_usage() {
        // Without stream_options, a streamed request returns no cached_tokens
        // and the cache metric silently reads zero.
        let req = ChatRequest::new("m", vec![ChatMessage::user("hi")]).streaming();
        assert!(req.stream);
        assert_eq!(req.stream_options.unwrap().include_usage, true);
    }

    #[test]
    fn a_schema_serializes_as_guided_json() {
        let req = ChatRequest::new("m", vec![ChatMessage::user("hi")])
            .with_schema(serde_json::json!({"type": "object"}));
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["guided_json"]["type"], "object");
    }

    #[test]
    fn optional_fields_are_omitted_rather_than_sent_as_null() {
        // Some OpenAI-compatible gateways reject explicit nulls.
        let req = ChatRequest::new("m", vec![ChatMessage::user("hi")]);
        let v = serde_json::to_value(&req).unwrap();
        for key in ["seed", "max_tokens", "guided_json", "stream_options"] {
            assert!(v.get(key).is_none(), "{key} was serialized");
        }
    }
}
