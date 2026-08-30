//! vLLM client.
//!
//! Spec: "Engine — vLLM, two processes — Prefix caching, speculation,
//! structured outputs, OpenAI-compatible API", and the non-goal "Writing an
//! inference engine. vLLM is the engine; the harness owns the control layer
//! above it."
//!
//! # Speculation is a server-launch setting, not a request parameter
//!
//! This is the one place the spec's model meets a hard vLLM constraint.
//! `--speculative-config` is fixed when a vLLM process starts; there is no
//! per-request field that switches a running server from DFlash2 to MTP. So
//! "every step has an explicit `(model, speculation, sampling)` triple" is
//! satisfied by *routing to the endpoint that was launched with that
//! speculation*, not by sending the speculation along with the request.
//!
//! [`EngineRegistry`] is that routing table. It maps
//! `(role, speculation method)` to an endpoint, which is why the spec's "two
//! processes" is a floor rather than a total: running Traceable (worker
//! DFlash2) and Deterministic (worker MTP) against the same deployment needs a
//! third process. A step whose endpoint is not registered is an error, in
//! keeping with "nothing defaults" — the harness will not quietly run an
//! MTP-planned step on a DFlash2 server and report acceptance numbers that mean
//! something else.

pub mod client;
pub mod metrics;
pub mod registry;
pub mod types;

pub use client::VllmClient;
pub use metrics::{AcceptanceSnapshot, scrape_acceptance};
pub use registry::{Endpoint, EngineRegistry, EngineRegistryError};
pub use types::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, EmbeddingResponse, PromptTokensDetails,
    Usage,
};
