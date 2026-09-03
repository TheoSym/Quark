//! The vLLM backend.

use crate::endpoint::{Endpoint, EngineKind, engine_typically_supports};
use crate::http::HttpClient;
use crate::metrics::{AcceptanceSnapshot, prometheus_lines};
use crate::types::{ChatRequest, ChatResponse, EmbeddingResponse};
use crate::{Engine, send_chat};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use qgi2_spec_types::Speculation;
use serde_json::json;

/// The flags every vLLM process under QGI-2 needs, whatever the speculation.
///
/// - `--enable-prefix-caching`: the whole premise.
/// - `--mamba-cache-mode align`: on a hybrid (GDN) model the recurrent state
///   resumes only at aligned boundaries; without this, prefix hits on the
///   spec's own planner/worker family do not happen at all
///   (syv-ai/qwen38-27b-rtx3090, `batch/start_qwen.sh`).
/// - `--enable-force-include-usage`: usage on every response, streamed
///   included. Without it `cached_tokens` is absent on streamed replies and
///   the cache metric reads zero -- which the spec calls a bug -- for a
///   prefix that was fine.
pub const COMMON_FLAGS: &str =
    "--enable-prefix-caching --mamba-cache-mode align --enable-force-include-usage";

pub struct VllmEngine {
    http: HttpClient,
}

impl VllmEngine {
    pub fn new(http: HttpClient) -> Self {
        Self { http }
    }

    /// The server's own prefix-cache hit rate, from
    /// `vllm:prefix_cache_hits / vllm:prefix_cache_queries`.
    ///
    /// The harness's per-request figure from `cached_tokens` only sees what
    /// this session reused. These counters also cover prefix shared *between*
    /// sessions on the same server, which is what running several personas
    /// against one process is for. vLLM never emits DeepSeek's
    /// `prompt_cache_hit_tokens`; these are the equivalent
    /// (syv-ai/qwen38-27b-rtx3090, docs/gotchas.md).
    ///
    /// Both are monotonic counters, so a lifetime reading is what you get here;
    /// difference two readings for a window.
    pub async fn cache_hit_counters(&self, endpoint: &Endpoint) -> Option<(u64, u64)> {
        let url = format!("{}/metrics", endpoint.root());
        let body = self
            .http
            .get_text(&url, endpoint.api_key.as_deref())
            .await
            .ok()?;
        let mut hits = None;
        let mut queries = None;
        for (name, v) in prometheus_lines(&body) {
            match name {
                "vllm:prefix_cache_hits" | "vllm:prefix_cache_hits_total" => {
                    hits = Some(v.max(0.0) as u64)
                }
                "vllm:prefix_cache_queries" | "vllm:prefix_cache_queries_total" => {
                    queries = Some(v.max(0.0) as u64)
                }
                _ => {}
            }
        }
        Some((hits?, queries?))
    }
}

#[async_trait]
impl Engine for VllmEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Vllm
    }

    fn supports(&self, speculation: Speculation) -> bool {
        engine_typically_supports(EngineKind::Vllm, speculation)
    }

    async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse> {
        let mut body = req
            .clone()
            .streaming(endpoint.stream)
            .with_cache_breakpoint(endpoint.cache_control)
            .to_openai_body(&endpoint.model);

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
            Speculation::Off => format!("{} (no speculation)", COMMON_FLAGS),
            // DSpark on vLLM takes the external drafter in the speculative
            // config (vllm #47808), not as a separate flag.
            Speculation::DSpark { n } => format!(
                "{} --speculative-config '{{\"method\":\"dspark\",\"model\":\"/path/to/RadixArk-Qwen3.8-27B-DSpark\",\"num_speculative_tokens\":{n}}}'",
                COMMON_FLAGS
            ),
            s if self.supports(s) => format!(
                "{} --speculative-config '{{\"method\":\"{}\",\"num_speculative_tokens\":{}}}'",
                COMMON_FLAGS,
                s.method_name(),
                s.lookahead()
            ),
            s => format!("vLLM does not implement {s}"),
        }
    }
}

/// Reject a request vLLM cannot serve before it is sent.
pub fn check_serviceable(endpoint: &Endpoint, spec: Speculation) -> Result<()> {
    if !engine_typically_supports(EngineKind::Vllm, spec) {
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
    fn every_vllm_hint_forces_usage_and_aligns_the_mamba_cache() {
        // Without --enable-force-include-usage, streamed replies carry no
        // cached_tokens and the cache metric reads zero for a healthy prefix.
        // Without --mamba-cache-mode align, a hybrid model never hits at all.
        for spec in [
            Speculation::Off,
            Speculation::Mtp { n: 3 },
            Speculation::DFlash2 { n: 7 },
            Speculation::DSpark { n: 7 },
        ] {
            let hint = engine().launch_hint(spec);
            assert!(hint.contains("--enable-force-include-usage"), "{spec}: {hint}");
            assert!(hint.contains("--mamba-cache-mode align"), "{spec}: {hint}");
            assert!(hint.contains("--enable-prefix-caching"), "{spec}: {hint}");
        }
    }

    #[test]
    fn dspark_on_vllm_names_the_external_drafter() {
        let hint = engine().launch_hint(Speculation::DSpark { n: 7 });
        assert!(hint.contains("\"method\":\"dspark\""), "{hint}");
        assert!(hint.contains("\"model\":"), "{hint}");
        assert!(hint.contains("\"num_speculative_tokens\":7"), "{hint}");
    }

    #[test]
    fn prefix_cache_counters_parse() {
        let body = "vllm:prefix_cache_queries_total 1000.0\nvllm:prefix_cache_hits_total 870.0\n";
        let mut hits = None;
        let mut queries = None;
        for (name, v) in prometheus_lines(body) {
            match name {
                "vllm:prefix_cache_hits_total" => hits = Some(v as u64),
                "vllm:prefix_cache_queries_total" => queries = Some(v as u64),
                _ => {}
            }
        }
        assert_eq!((hits, queries), (Some(870), Some(1000)));
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
