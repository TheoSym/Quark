//! The vLLM backend.

use crate::endpoint::{Endpoint, EngineKind, engine_supports};
use crate::http::HttpClient;
use crate::metrics::{AcceptanceSnapshot, prometheus_lines};
use crate::types::{ChatRequest, ChatResponse, EmbeddingResponse};
use crate::Engine;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use qgi2_spec_types::Speculation;
use serde_json::json;

pub struct VllmEngine {
    http: HttpClient,
}

impl VllmEngine {
    pub fn new(http: HttpClient) -> Self {
        Self { http }
    }
}

#[async_trait]
impl Engine for VllmEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Vllm
    }

    fn supports(&self, speculation: Speculation) -> bool {
        engine_supports(EngineKind::Vllm, speculation)
    }

    async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse> {
        let mut body = req.to_openai_body(&endpoint.model);

        // vLLM's guided decoding is a top-level `guided_json` field.
        if let Some(schema) = &req.schema {
            body.insert("guided_json".into(), schema.clone());
            if req.sampling.seed.is_some() {
                // Pin the backend when the profile demands reproducibility:
                // guided backends can differ in which token they admit first at
                // a grammar branch, which is enough to diverge two "identical"
                // greedy runs.
                body.insert("guided_decoding_backend".into(), json!("xgrammar"));
            }
        }

        let text = self
            .http
            .post_json(endpoint, "/chat/completions", &body)
            .await?;
        serde_json::from_str(&text)
            .with_context(|| format!("decoding chat response from {}", endpoint.base_url))
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

        let mut draft = 0u64;
        let mut accepted = 0u64;
        let mut steps = 0u64;
        let mut seen = false;

        for (name, value) in prometheus_lines(&body) {
            let v = if value >= 0.0 { value as u64 } else { continue };
            match name {
                "vllm:spec_decode_num_draft_tokens_total" | "vllm:spec_decode_num_draft_tokens" => {
                    draft = draft.max(v);
                    seen = true;
                }
                "vllm:spec_decode_num_accepted_tokens_total"
                | "vllm:spec_decode_num_accepted_tokens" => {
                    accepted = accepted.max(v);
                    seen = true;
                }
                "vllm:spec_decode_num_drafts_total" | "vllm:spec_decode_num_drafts" => {
                    steps = steps.max(v);
                    seen = true;
                }
                _ => {}
            }
        }

        Ok(if seen {
            AcceptanceSnapshot::Counters {
                draft_tokens: draft,
                accepted_tokens: accepted,
                num_steps: steps,
            }
        } else {
            AcceptanceSnapshot::Unavailable
        })
    }

    async fn health(&self, endpoint: &Endpoint) -> bool {
        self.http.health(endpoint).await
    }

    fn launch_hint(&self, speculation: Speculation) -> String {
        match speculation {
            Speculation::Off => "--enable-prefix-caching (no speculation)".into(),
            s if self.supports(s) => format!(
                "--enable-prefix-caching --speculative-config '{{\"method\":\"{}\",\"num_speculative_tokens\":{}}}'",
                s.method_name(),
                s.lookahead()
            ),
            s => format!("vLLM does not implement {s}"),
        }
    }
}

/// Reject a request vLLM cannot serve before it is sent.
pub fn check_serviceable(endpoint: &Endpoint, spec: Speculation) -> Result<()> {
    if !engine_supports(EngineKind::Vllm, spec) {
        bail!("vLLM does not implement {spec} (endpoint {})", endpoint.base_url);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest};
    use qgi2_spec_types::{Profile, Sampling};

    fn engine() -> VllmEngine {
        VllmEngine::new(HttpClient::default())
    }

    #[test]
    fn a_schema_becomes_guided_json() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_schema(json!({"type": "object"}));
        let mut body = req.to_openai_body("m");
        if let Some(s) = &req.schema {
            body.insert("guided_json".into(), s.clone());
        }
        assert_eq!(body["guided_json"]["type"], "object");
    }

    #[test]
    fn a_deterministic_request_pins_the_guided_backend() {
        // Guided backends can differ in which token they admit first at a
        // grammar branch, which diverges two otherwise identical greedy runs.
        let s = Profile::Deterministic.apply_sampling(Sampling::at_temperature(0.7));
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_sampling(s)
            .with_schema(json!({"type": "object"}));
        assert!(req.sampling.seed.is_some());
    }

    #[test]
    fn acceptance_parses_vllms_counters() {
        let body = "\
vllm:spec_decode_num_draft_tokens_total{model=\"w\"} 7000.0
vllm:spec_decode_num_accepted_tokens_total{model=\"w\"} 5000.0
vllm:spec_decode_num_drafts_total{model=\"w\"} 1000.0
vllm:num_requests_running 2.0
";
        let mut draft = 0u64;
        let mut accepted = 0u64;
        let mut steps = 0u64;
        for (name, value) in prometheus_lines(body) {
            match name {
                "vllm:spec_decode_num_draft_tokens_total" => draft = value as u64,
                "vllm:spec_decode_num_accepted_tokens_total" => accepted = value as u64,
                "vllm:spec_decode_num_drafts_total" => steps = value as u64,
                _ => {}
            }
        }
        let snap = AcceptanceSnapshot::Counters {
            draft_tokens: draft,
            accepted_tokens: accepted,
            num_steps: steps,
        };
        assert_eq!(snap.tokens_per_step(), Some(6.0));
    }

    #[test]
    fn the_launch_hint_is_a_flag_you_can_paste() {
        let hint = engine().launch_hint(Speculation::DFlash2 { n: 7 });
        assert!(hint.contains("--speculative-config"), "{hint}");
        assert!(hint.contains("dflash2"), "{hint}");
        assert!(hint.contains("7"), "{hint}");
    }

    #[test]
    fn the_hint_says_so_when_vllm_cannot_serve_it() {
        assert!(
            engine()
                .launch_hint(Speculation::Eagle3 { n: 5 })
                .contains("does not implement")
        );
    }
}
