//! Endpoint routing: which process serves a given `(role, speculation)`.
//!
//! Speculation is fixed when an inference server starts — `--speculative-config`
//! on vLLM, `--speculative-algorithm` on SGLang — and no per-request field
//! changes it. So the spec's "explicit speculation per step" is satisfied by
//! sending the step to the process that was launched with it.
//!
//! A missing endpoint is an error rather than a fallback. Running an
//! MTP-planned step on a DFlash2 server would produce acceptance numbers
//! describing a different configuration than the router logged, and the spec
//! treats those numbers as a correctness signal.

use qgi2_spec_types::{ModelRole, Speculation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Which inference server software an endpoint runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Vllm,
    Sglang,
}

impl EngineKind {
    pub const ALL: [EngineKind; 2] = [EngineKind::Vllm, EngineKind::Sglang];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for EngineKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vllm" => Ok(Self::Vllm),
            "sglang" | "sgl" => Ok(Self::Sglang),
            other => Err(format!("unknown engine {other:?}; expected vllm or sglang")),
        }
    }
}

/// One inference server process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Base URL including the `/v1` suffix, e.g. `http://127.0.0.1:8000/v1`.
    pub base_url: String,
    /// The model name this process serves.
    pub model: String,
    /// Which server software it runs.
    #[serde(default = "default_engine")]
    pub engine: EngineKind,
    /// The speculation method it was launched with.
    pub speculation_method: String,
    /// Lookahead it was launched with, for cross-checking against the router.
    pub speculation_n: u8,
    /// Optional bearer token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

fn default_engine() -> EngineKind {
    EngineKind::Vllm
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
            engine: EngineKind::Vllm,
            speculation_method: speculation.method_name().to_string(),
            speculation_n: speculation.lookahead(),
            api_key: None,
        }
    }

    pub fn with_engine(mut self, engine: EngineKind) -> Self {
        self.engine = engine;
        self
    }

    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    /// Whether this process implements the speculation a step asked for.
    ///
    /// The lookahead must match too: acceptance floors are stated per method
    /// *and* depth, so an EAGLE3 n=3 process cannot stand in for the n=5 the
    /// router planned.
    pub fn serves(&self, spec: Speculation) -> bool {
        self.speculation_method == spec.method_name() && self.speculation_n == spec.lookahead()
    }

    /// The speculation this endpoint claims, as a value.
    pub fn speculation(&self) -> Option<Speculation> {
        let n = self.speculation_n;
        Some(match self.speculation_method.as_str() {
            "mtp" => Speculation::Mtp { n },
            "dflash2" => Speculation::DFlash2 { n },
            "eagle3" => Speculation::Eagle3 { n },
            "ngram" => Speculation::NGram { n },
            "off" | "none" => Speculation::Off,
            _ => return None,
        })
    }

    pub fn url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    /// The server root, without the `/v1` suffix. Metrics live there.
    pub fn root(&self) -> String {
        self.base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineRegistryError {
    #[error(
        "no endpoint registered for the {role} with {speculation}. \
         Launch a server with that speculation and register it, or choose a profile \
         whose speculation you have deployed. Registered: [{registered}]"
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
    /// The embedding endpoint. Optional because the Quick profile never calls
    /// it.
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

    /// Endpoints whose declared speculation is unusual for their engine.
    ///
    /// Advisory only. See [`engine_typically_supports`] for why this cannot be
    /// an error.
    pub fn unusual_pairings(&self) -> Vec<String> {
        self.all()
            .iter()
            .filter_map(|(role, e)| {
                let spec = e.speculation()?;
                (!engine_typically_supports(e.engine, spec)).then(|| {
                    format!(
                        "{role} at {} declares {spec} on {}, which is not one of that \
                         engine's usual speculators — fine if your build serves it (the QGI \
                         fleet runs DFlash2 on SGLang), worth a second look if you did not \
                         intend it",
                        e.base_url, e.engine
                    )
                })
            })
            .collect()
    }

    /// What is registered for a role, for the error message.
    pub fn describe(&self, role: ModelRole) -> String {
        self.endpoints
            .get(role.as_str())
            .map(|m| {
                m.values()
                    .map(|e| {
                        format!(
                            "{} {} n={} at {}",
                            e.engine, e.speculation_method, e.speculation_n, e.base_url
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default()
    }

    /// Every registered endpoint, for a startup summary and metrics scraping.
    pub fn all(&self) -> Vec<(&str, &Endpoint)> {
        self.endpoints
            .iter()
            .flat_map(|(role, m)| m.values().map(move |e| (role.as_str(), e)))
            .collect()
    }

    /// Every distinct engine kind in use.
    pub fn engine_kinds(&self) -> Vec<EngineKind> {
        let mut kinds: Vec<EngineKind> = self
            .all()
            .iter()
            .map(|(_, e)| e.engine)
            .chain(self.embedder.iter().map(|e| e.engine))
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        kinds
    }

    /// Check every triple a set of plans would ask for, before a session starts.
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

/// Which speculators an engine *typically* ships.
///
/// **Advisory, never a veto.** An earlier version treated this as ground truth
/// and refused to route a step whose speculation was not in the table. That was
/// wrong in the way guardrails usually are: the QGI fleet runs Qwen3.8-27B with
/// DFlash2 spec-decode on SGLang, which this table said was impossible, so the
/// harness would have hard-errored on a live, working deployment.
///
/// The endpoint's declared configuration is ground truth — it describes what
/// was actually launched. This table only powers a warning, because engines
/// gain speculators between releases and a deployment can be patched, and
/// neither is something a compile-time list can know about.
pub fn engine_typically_supports(engine: EngineKind, spec: Speculation) -> bool {
    match engine {
        EngineKind::Vllm => matches!(
            spec,
            Speculation::Mtp { .. }
                | Speculation::DFlash2 { .. }
                | Speculation::NGram { .. }
                | Speculation::Off
        ),
        // SGLang's speculators are EAGLE/EAGLE3, NEXTN (which is MTP for models
        // that ship an MTP head), and n-gram. DFlash2 is not among them.
        EngineKind::Sglang => matches!(
            spec,
            Speculation::Eagle3 { .. }
                | Speculation::Mtp { .. }
                | Speculation::NGram { .. }
                | Speculation::Off
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vllm_registry() -> EngineRegistry {
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

    fn sglang_registry() -> EngineRegistry {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://onyxtron-g12:30000/v1", "planner", Speculation::Mtp { n: 2 })
                .with_engine(EngineKind::Sglang),
        );
        r.register(
            ModelRole::Worker,
            Endpoint::new("http://rhoditron-g24:30000/v1", "worker", Speculation::Eagle3 { n: 5 })
                .with_engine(EngineKind::Sglang),
        );
        r
    }

    #[test]
    fn a_registered_triple_resolves() {
        let e = vllm_registry()
            .resolve(ModelRole::Worker, Speculation::DFlash2 { n: 7 })
            .unwrap()
            .clone();
        assert_eq!(e.model, "worker");
        assert_eq!(e.engine, EngineKind::Vllm);
    }

    #[test]
    fn an_sglang_worker_resolves_on_eagle3() {
        let e = sglang_registry()
            .resolve(ModelRole::Worker, Speculation::Eagle3 { n: 5 })
            .unwrap()
            .clone();
        assert_eq!(e.engine, EngineKind::Sglang);
    }

    #[test]
    fn an_sglang_endpoint_may_declare_dflash2() {
        // The QGI fleet runs Qwen3.8-27B with DFlash2 spec-decode on SGLang. An
        // earlier version vetoed this pairing from a hardcoded table and would
        // have refused to route against a live, working deployment. The
        // endpoint's declaration is ground truth.
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Worker,
            Endpoint::new("http://100.107.254.57:18031/v1", "qwen3.8-27b", Speculation::DFlash2 { n: 7 })
                .with_engine(EngineKind::Sglang),
        );
        let e = r
            .resolve(ModelRole::Worker, Speculation::DFlash2 { n: 7 })
            .expect("a declared DFlash2 SGLang endpoint must route");
        assert_eq!(e.engine, EngineKind::Sglang);
    }

    #[test]
    fn an_unusual_pairing_is_a_warning_not_a_refusal() {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Worker,
            Endpoint::new("http://h/v1", "m", Speculation::DFlash2 { n: 7 })
                .with_engine(EngineKind::Sglang),
        );
        assert!(r.resolve(ModelRole::Worker, Speculation::DFlash2 { n: 7 }).is_ok());
        let warnings = r.unusual_pairings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("worth a second look"), "{warnings:?}");
    }

    #[test]
    fn a_missing_but_supported_speculation_is_a_registration_error() {
        // vLLM *can* do MTP n=3; it just isn't deployed. Different fix, so a
        // different error.
        let err = vllm_registry()
            .resolve(ModelRole::Worker, Speculation::Mtp { n: 3 })
            .unwrap_err();
        assert!(matches!(err, EngineRegistryError::NoEndpoint { .. }), "{err:?}");
        assert!(err.to_string().contains("dflash2 n=7"), "lists what is registered");
    }

    #[test]
    fn the_lookahead_must_match_too() {
        assert!(
            vllm_registry()
                .resolve(ModelRole::Worker, Speculation::DFlash2 { n: 3 })
                .is_err()
        );
    }

    #[test]
    fn engine_support_tables_differ_where_it_matters() {
        assert!(engine_typically_supports(EngineKind::Vllm, Speculation::DFlash2 { n: 7 }));
        // Advisory only — the QGI fleet does exactly this pairing in production.
        assert!(!engine_typically_supports(EngineKind::Sglang, Speculation::DFlash2 { n: 7 }));
        assert!(engine_typically_supports(EngineKind::Sglang, Speculation::Eagle3 { n: 5 }));
        assert!(!engine_typically_supports(EngineKind::Vllm, Speculation::Eagle3 { n: 5 }));
        // Both do MTP and n-gram.
        for k in EngineKind::ALL {
            assert!(engine_typically_supports(k, Speculation::Mtp { n: 2 }));
            assert!(engine_typically_supports(k, Speculation::NGram { n: 4 }));
            assert!(engine_typically_supports(k, Speculation::Off));
        }
    }

    #[test]
    fn url_joining_survives_a_trailing_slash() {
        let e = Endpoint::new("http://h:8000/v1/", "m", Speculation::Off);
        assert_eq!(e.url("/chat/completions"), "http://h:8000/v1/chat/completions");
        assert_eq!(e.url("chat/completions"), "http://h:8000/v1/chat/completions");
        assert_eq!(e.root(), "http://h:8000");
    }

    #[test]
    fn engine_kinds_lists_what_is_deployed() {
        let mut r = sglang_registry();
        r.set_embedder(
            Endpoint::new("http://h:8002/v1", "embed", Speculation::Off)
                .with_engine(EngineKind::Vllm),
        );
        assert_eq!(r.engine_kinds(), vec![EngineKind::Vllm, EngineKind::Sglang]);
    }

    #[test]
    fn the_registry_round_trips_through_config() {
        let r = sglang_registry();
        let back: EngineRegistry = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        let e = back
            .resolve(ModelRole::Worker, Speculation::Eagle3 { n: 5 })
            .unwrap();
        assert_eq!(e.engine, EngineKind::Sglang);
    }

    #[test]
    fn an_endpoint_reports_the_speculation_it_claims() {
        let e = Endpoint::new("http://h/v1", "m", Speculation::Eagle3 { n: 5 });
        assert_eq!(e.speculation(), Some(Speculation::Eagle3 { n: 5 }));
    }
}
