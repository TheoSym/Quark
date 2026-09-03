//! The inference engine abstraction, with vLLM and SGLang backends.
//!
//! The spec names vLLM as the engine. In practice a fleet may run something
//! else — SGLang, in particular, whose RadixAttention is prefix sharing across
//! requests and therefore an unusually good match for a harness whose whole
//! premise is a large byte-stable prefix.
//!
//! Everything above this crate is engine-agnostic: the assembler, router,
//! rules, and graph never learn which server answered. What differs between
//! engines is narrow and entirely contained in [`Engine`]:
//!
//! | | vLLM | SGLang |
//! |---|---|---|
//! | Schema constraint | `guided_json` | `response_format: {type: json_schema, …}` |
//! | Cached tokens | `usage.prompt_tokens_details.cached_tokens` | same, or a top-level `cached_tokens` on older builds |
//! | Acceptance | derived from `vllm:spec_decode_*` counters | `sglang:spec_accept_length` reports it directly |
//! | Cache hit rate | computed per request | `sglang:cache_hit_rate` reports it directly |
//! | Speculators | MTP, DFlash2, n-gram | EAGLE/EAGLE3, NEXTN (MTP), n-gram — **no DFlash2** |
//!
//! That last row has a consequence worth stating plainly: the spec's Traceable
//! profile calls for `DFlash2 n=7` on the worker, and no SGLang deployment can
//! serve it. [`Engine::supports`] is how that becomes a startup error naming
//! the mismatch rather than a silent substitution.

pub mod endpoint;
pub mod hicache;
pub mod http;
pub mod metrics;
pub mod scripted;
pub mod sglang;
pub mod types;
pub mod vllm;

pub use endpoint::{Endpoint, EngineKind, EngineRegistry, EngineRegistryError, engine_typically_supports};
pub use hicache::{
    GdnStatePool, HiCacheConfig, HiCacheStats, L2Sizing, L3Backend, PageAlignment, Tier,
    VramBudget, scrape_hicache,
};
pub use http::HttpClient;
pub use metrics::AcceptanceSnapshot;
pub use scripted::{Script, ScriptedEngine, SeenStep};
pub use sglang::SglangEngine;
pub use types::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, EmbeddingResponse, PromptTokensDetails,
    Usage,
};
pub use vllm::VllmEngine;

use anyhow::Result;
use async_trait::async_trait;
use qgi2_spec_types::Speculation;
use std::sync::Arc;

/// One inference backend.
///
/// Implementations are stateless beyond an HTTP client: the endpoint is passed
/// per call, because one engine kind typically serves several processes (a
/// planner and a worker, at minimum).
#[async_trait]
pub trait Engine: Send + Sync {
    /// Stable identifier, matching the config's `engine =` value.
    fn kind(&self) -> EngineKind;

    /// Whether this engine can run a speculation method at all.
    ///
    /// Distinct from whether a *particular process* was launched with it — that
    /// is [`EngineRegistry::resolve`]'s job. This answers the earlier question
    /// of whether asking is even coherent, so a profile that cannot be served
    /// fails with "SGLang has no DFlash2" rather than "no endpoint registered",
    /// which would send you looking for a config mistake that isn't there.
    fn supports(&self, speculation: Speculation) -> bool;

    /// Send a chat completion.
    async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse>;

    /// Embed texts. Spec: entry-point retrieval only.
    async fn embed(&self, endpoint: &Endpoint, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Speculation acceptance for this process, scraped from its metrics.
    async fn acceptance(&self, endpoint: &Endpoint) -> Result<AcceptanceSnapshot>;

    /// Whether the process is up, for the startup preflight.
    async fn health(&self, endpoint: &Endpoint) -> bool;

    /// The engine's launch flag for a speculation, for error messages and docs.
    ///
    /// Telling someone "no endpoint serves mtp n=2" is only half an answer; the
    /// other half is the flag they need to launch one.
    fn launch_hint(&self, speculation: Speculation) -> String;
}

/// Send a prepared chat body, streaming when the endpoint asks for it.
///
/// Shared by both backends: the wire body differs (guided_json vs
/// response_format), the transport does not.
pub async fn send_chat(
    http: &HttpClient,
    endpoint: &Endpoint,
    body: &serde_json::Map<String, serde_json::Value>,
) -> Result<ChatResponse> {
    use anyhow::Context;

    if !body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        let text = http.post_json(endpoint, "/chat/completions", body).await?;
        return serde_json::from_str(&text)
            .with_context(|| format!("decoding chat response from {}", endpoint.base_url));
    }

    let (text, usage) = http.post_sse(endpoint, "/chat/completions", body).await?;
    // A streamed response is reassembled into the same shape a non-streamed one
    // would have had, so nothing above this line knows the difference.
    let usage = usage.and_then(|u| serde_json::from_value::<Usage>(u).ok());
    Ok(ChatResponse {
        id: String::new(),
        model: body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant(text),
            finish_reason: Some("stop".into()),
        }],
        usage,
    })
}

/// Build the engine for a kind.
pub fn engine_for(kind: EngineKind, http: HttpClient) -> Arc<dyn Engine> {
    match kind {
        EngineKind::Vllm => Arc::new(VllmEngine::new(http)),
        EngineKind::Sglang => Arc::new(SglangEngine::new(http)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_serves_the_specs_speculators() {
        let e = engine_for(EngineKind::Vllm, HttpClient::default());
        assert!(e.supports(Speculation::Mtp { n: 2 }));
        assert!(e.supports(Speculation::DFlash2 { n: 7 }));
        assert!(e.supports(Speculation::NGram { n: 4 }));
        assert!(e.supports(Speculation::Off));
    }

    #[test]
    fn sglang_has_no_dflash2() {
        // The spec's Traceable profile asks for DFlash2 n=7 on the worker. No
        // SGLang build can serve that, and saying so is more useful than an
        // endpoint-not-found error that sends someone hunting a config typo.
        let e = engine_for(EngineKind::Sglang, HttpClient::default());
        assert!(!e.supports(Speculation::DFlash2 { n: 7 }));
        assert!(e.supports(Speculation::Mtp { n: 3 }));
        assert!(e.supports(Speculation::Eagle3 { n: 5 }));
        assert!(e.supports(Speculation::Off));
    }

    #[test]
    fn vllm_has_no_eagle3() {
        // Symmetrically: EAGLE3 is SGLang's headline speculator and is not how
        // vLLM's spec config is expressed here.
        let e = engine_for(EngineKind::Vllm, HttpClient::default());
        assert!(!e.supports(Speculation::Eagle3 { n: 5 }));
    }

    #[test]
    fn every_engine_explains_how_to_launch_what_it_supports() {
        for kind in [EngineKind::Vllm, EngineKind::Sglang] {
            let e = engine_for(kind, HttpClient::default());
            for spec in [
                Speculation::Mtp { n: 2 },
                Speculation::DFlash2 { n: 7 },
                Speculation::Eagle3 { n: 5 },
                Speculation::NGram { n: 4 },
                Speculation::Off,
            ] {
                if e.supports(spec) {
                    let hint = e.launch_hint(spec);
                    assert!(
                        hint.contains("--"),
                        "{kind:?} gave no launch flag for {spec}: {hint}"
                    );
                }
            }
        }
    }
}
