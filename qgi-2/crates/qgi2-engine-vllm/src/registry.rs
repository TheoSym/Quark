//! Endpoint routing: which vLLM process serves a given `(role, speculation)`.
//!
//! See the crate docs for why this exists. In short: `--speculative-config` is
//! fixed at vLLM launch, so the harness satisfies "every step has an explicit
//! speculation" by sending the step to the process that was launched with it.
//!
//! A missing endpoint is an error rather than a fallback. Silently running an
//! MTP-planned step on a DFlash2 server would produce acceptance numbers that
//! describe a different configuration than the one the router logged — and the
//! spec treats those numbers as a correctness signal, not decoration.

use qgi2_spec_types::{ModelRole, Speculation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One vLLM process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Base URL including the `/v1` suffix, e.g. `http://127.0.0.1:8000/v1`.
    pub base_url: String,
    /// The model name this process serves.
    pub model: String,
    /// The speculation method it was launched with.
    pub speculation_method: String,
    /// Lookahead it was launched with, for cross-checking against the router.
    pub speculation_n: u8,
    /// Optional bearer token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl Endpoint {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        speculation: Speculation,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            speculation_method: speculation.method_name().to_string(),
            speculation_n: speculation.lookahead(),
            api_key: None,
        }
    }

    /// Whether this process actually implements the speculation a step asked
    /// for.
    ///
    /// The lookahead must match too: acceptance floors in the spec's success
    /// metrics are stated per method *and* depth, so a DFlash2 n=3 process
    /// cannot stand in for the n=7 the router planned.
    pub fn serves(&self, spec: Speculation) -> bool {
        self.speculation_method == spec.method_name() && self.speculation_n == spec.lookahead()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineRegistryError {
    #[error(
        "no vLLM endpoint registered for the {role} with {speculation}. \
         Launch a vLLM process with that --speculative-config and register it, \
         or choose a profile whose speculation you have deployed. \
         Registered: [{registered}]"
    )]
    NoEndpoint {
        role: ModelRole,
        speculation: Speculation,
        registered: String,
    },
    #[error(
        "endpoint {url} is registered for the {role} as {claimed}, but a step planned {planned}"
    )]
    Mismatch {
        url: String,
        role: ModelRole,
        claimed: String,
        planned: String,
    },
}

/// The `(role, speculation) -> endpoint` table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineRegistry {
    /// Keyed by `role` then `"method:n"`, so the config file is readable.
    endpoints: BTreeMap<String, BTreeMap<String, Endpoint>>,
    /// The embedding endpoint. Spec: "Qwen3-Embedding-0.6B via vLLM
    /// `/v1/embeddings` (MiniLM fallback)". Optional because the Quick profile
    /// never calls it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<Endpoint>,
}

fn spec_key(s: Speculation) -> String {
    format!("{}:{}", s.method_name(), s.lookahead())
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, role: ModelRole, endpoint: Endpoint) {
        let key = format!("{}:{}", endpoint.speculation_method, endpoint.speculation_n);
        self.endpoints
            .entry(role.as_str().to_string())
            .or_default()
            .insert(key, endpoint);
    }

    pub fn set_embedder(&mut self, endpoint: Endpoint) {
        self.embedder = Some(endpoint);
    }

    /// The endpoint that serves this step's triple.
    pub fn resolve(
        &self,
        role: ModelRole,
        speculation: Speculation,
    ) -> Result<&Endpoint, EngineRegistryError> {
        let by_spec = self.endpoints.get(role.as_str());
        let found = by_spec.and_then(|m| m.get(&spec_key(speculation)));

        match found {
            Some(e) if e.serves(speculation) => Ok(e),
            Some(e) => Err(EngineRegistryError::Mismatch {
                url: e.base_url.clone(),
                role,
                claimed: format!("{} n={}", e.speculation_method, e.speculation_n),
                planned: speculation.to_string(),
            }),
            None => Err(EngineRegistryError::NoEndpoint {
                role,
                speculation,
                registered: self.describe(role),
            }),
        }
    }

    /// What is registered for a role, for the error message.
    pub fn describe(&self, role: ModelRole) -> String {
        self.endpoints
            .get(role.as_str())
            .map(|m| {
                m.values()
                    .map(|e| format!("{} n={} at {}", e.speculation_method, e.speculation_n, e.base_url))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default()
    }

    /// Every registered endpoint, for a startup summary and for metrics
    /// scraping.
    pub fn all(&self) -> Vec<(&str, &Endpoint)> {
        self.endpoints
            .iter()
            .flat_map(|(role, m)| m.values().map(move |e| (role.as_str(), e)))
            .collect()
    }

    /// Check every triple a set of profiles would ask for, before a session
    /// starts.
    ///
    /// Discovering a missing endpoint on turn one of a long session is worse
    /// than discovering it at startup, so the binary calls this eagerly.
    pub fn preflight(
        &self,
        plans: &[qgi2_spec_types::StepPlan],
    ) -> Result<(), EngineRegistryError> {
        for p in plans {
            self.resolve(p.role, p.speculation)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> EngineRegistry {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://127.0.0.1:8000/v1", "planner", Speculation::Mtp { n: 2 }),
        );
        r.register(
            ModelRole::Worker,
            Endpoint::new("http://127.0.0.1:8001/v1", "worker", Speculation::DFlash2 { n: 7 }),
        );
        r
    }

    #[test]
    fn a_registered_triple_resolves() {
        let r = registry();
        let e = r.resolve(ModelRole::Worker, Speculation::DFlash2 { n: 7 }).unwrap();
        assert_eq!(e.model, "worker");
    }

    #[test]
    fn a_missing_speculation_is_an_error_not_a_fallback() {
        // The Deterministic profile needs a worker launched with MTP. Running
        // it on the DFlash2 process would report acceptance for a different
        // configuration than the router logged.
        let r = registry();
        let err = r
            .resolve(ModelRole::Worker, Speculation::Mtp { n: 3 })
            .unwrap_err();
        assert!(matches!(err, EngineRegistryError::NoEndpoint { .. }));
        let msg = err.to_string();
        assert!(msg.contains("--speculative-config"), "{msg}");
        assert!(msg.contains("dflash2 n=7"), "error lists what is registered: {msg}");
    }

    #[test]
    fn the_lookahead_must_match_too() {
        // Acceptance floors are stated per method and depth, so n=3 cannot
        // stand in for n=7.
        let r = registry();
        assert!(r.resolve(ModelRole::Worker, Speculation::DFlash2 { n: 3 }).is_err());
    }

    #[test]
    fn preflight_catches_a_missing_endpoint_before_the_session_starts() {
        use qgi2_spec_types::{Persona, Profile, Mood};
        let r = registry();

        let traceable = qgi2_router_plans(Mood::Builder, Profile::Traceable);
        assert!(r.preflight(&traceable).is_ok(), "{:?}", r.preflight(&traceable));

        // Deterministic routes the worker to MTP n=3, which is not registered.
        let deterministic = qgi2_router_plans(Mood::Builder, Profile::Deterministic);
        assert!(r.preflight(&deterministic).is_err());

        let _ = Persona::default();
    }

    /// Local stand-in so this crate does not depend on qgi2-router just for a
    /// test. Mirrors the router's role and speculation choices.
    fn qgi2_router_plans(
        _mood: qgi2_spec_types::Mood,
        profile: qgi2_spec_types::Profile,
    ) -> Vec<qgi2_spec_types::StepPlan> {
        use qgi2_spec_types::{Sampling, StepKind};
        vec![
            qgi2_spec_types::StepPlan {
                step: StepKind::Plan,
                role: ModelRole::Planner,
                speculation: Speculation::Mtp { n: 2 },
                sampling: Sampling::at_temperature(0.3),
                schema: Some(serde_json::json!({"type": "object"})),
            },
            qgi2_spec_types::StepPlan {
                step: StepKind::Extract,
                role: ModelRole::Worker,
                speculation: profile.worker_speculation(),
                sampling: Sampling::at_temperature(0.3),
                schema: Some(serde_json::json!({"type": "object"})),
            },
        ]
    }

    #[test]
    fn url_joining_survives_a_trailing_slash() {
        let e = Endpoint::new("http://h:8000/v1/", "m", Speculation::Off);
        assert_eq!(e.url("/chat/completions"), "http://h:8000/v1/chat/completions");
        assert_eq!(e.url("chat/completions"), "http://h:8000/v1/chat/completions");
    }

    #[test]
    fn the_registry_round_trips_through_config() {
        let r = registry();
        let json = serde_json::to_string(&r).unwrap();
        let back: EngineRegistry = serde_json::from_str(&json).unwrap();
        assert!(back.resolve(ModelRole::Planner, Speculation::Mtp { n: 2 }).is_ok());
    }
}
