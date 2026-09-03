//! Individual steps: running one model call under its plan, and parsing the
//! structured result.
//!
//! Every structured step goes through [`run_structured`], which hands the
//! step's schema to the engine and then *still* validates the shape it got
//! back. Guided decoding makes malformed output very unlikely, not impossible:
//! a deployment can have the guided backend misconfigured, or be an engine
//! build that ignores the constraint field entirely. The failure mode without a
//! check here is a silently empty extraction that looks like "the model found
//! nothing".

use anyhow::{Context, Result, bail};
use qgi2_engine::{ChatMessage, ChatRequest, ChatResponse, Engine, EngineRegistry};
use std::sync::Arc;
use qgi2_spec_types::{ProposedFact, StepPlan};
use serde::Deserialize;
use serde_json::Value;

/// The plan step's output.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PlanOutput {
    #[serde(default)]
    pub steps: Vec<PlanStep>,
    #[serde(default)]
    pub needs_tools: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlanStep {
    pub intent: String,
    #[serde(default)]
    pub tool: Option<String>,
}

/// The extract step's output.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExtractOutput {
    #[serde(default)]
    pub facts: Vec<ProposedFact>,
}

/// The route step's output.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RouteOutput {
    #[serde(default)]
    pub entry_points: Vec<String>,
    #[serde(default)]
    pub suggested_mood: Option<String>,
}

/// The tool-args step's output.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolArgsOutput {
    pub tool: String,
    pub arguments: Value,
}

/// Run one model step and return the raw response.
///
/// The endpoint is resolved from the plan's `(role, speculation)` pair — see
/// [`qgi2_engine::EngineRegistry`] for why speculation selects a process
/// rather than a request field.
pub async fn run_step(
    engines: &Engines,
    registry: &EngineRegistry,
    plan: &StepPlan,
    system: &str,
    user: &str,
) -> Result<ChatResponse> {
    let endpoint = registry
        .resolve(plan.role, plan.speculation)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut req = ChatRequest::new(vec![ChatMessage::system(system), ChatMessage::user(user)])
        .with_sampling(plan.sampling);

    if let Some(schema) = &plan.schema {
        req = req.with_schema(schema.clone());
    }

    let engine = engines.for_endpoint(endpoint)?;
    engine
        .chat(endpoint, &req)
        .await
        .with_context(|| format!("{} step on the {}", plan.step, plan.role))
}

/// The engine backends a session can reach, one per kind in its registry.
///
/// A session's endpoints may span engines — an SGLang worker beside a vLLM
/// planner is a reasonable deployment — so the backend is chosen per endpoint
/// rather than once per session.
#[derive(Clone, Default)]
pub struct Engines {
    by_kind: std::collections::BTreeMap<qgi2_engine::EngineKind, Arc<dyn Engine>>,
}

impl Engines {
    /// Build one backend per engine kind the registry mentions.
    pub fn for_registry(registry: &EngineRegistry) -> Self {
        let http = qgi2_engine::HttpClient::default();
        let by_kind = registry
            .engine_kinds()
            .into_iter()
            .map(|k| (k, qgi2_engine::engine_for(k, http.clone())))
            .collect();
        Self { by_kind }
    }

    /// Install a backend for a kind, replacing whatever `for_registry` built.
    ///
    /// This is how a scripted engine reaches the loop: register endpoints as
    /// usual, then swap the kind they declare for one that answers from a
    /// script. Everything above -- routing, assembly, rules, metrics -- sees
    /// exactly what it would see against a live server.
    pub fn with_engine(mut self, kind: qgi2_engine::EngineKind, engine: Arc<dyn Engine>) -> Self {
        self.by_kind.insert(kind, engine);
        self
    }

    pub fn for_endpoint(&self, endpoint: &qgi2_engine::Endpoint) -> Result<&Arc<dyn Engine>> {
        self.by_kind.get(&endpoint.engine).ok_or_else(|| {
            anyhow::anyhow!(
                "no backend built for engine {} at {}",
                endpoint.engine,
                endpoint.base_url
            )
        })
    }

    pub fn kinds(&self) -> Vec<qgi2_engine::EngineKind> {
        self.by_kind.keys().copied().collect()
    }
}

impl std::fmt::Debug for Engines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engines").field("kinds", &self.kinds()).finish()
    }
}

/// Run a structured step, deserialize its output, and hand back the response.
///
/// The response comes back alongside the parsed value on purpose: its `usage`
/// carries `cached_tokens`, and a helper that discarded it would leave every
/// structured step invisible to the cache-hit metric — which the spec treats as
/// a correctness signal, not a nice-to-have.
///
/// Fails loudly on unparseable output rather than returning a default: a step
/// that silently yields `Default` is indistinguishable from one that genuinely
/// found nothing, and the difference matters for the rejection-rate metric.
pub async fn run_structured<T: for<'de> Deserialize<'de>>(
    engines: &Engines,
    registry: &EngineRegistry,
    plan: &StepPlan,
    system: &str,
    user: &str,
) -> Result<(T, ChatResponse)> {
    if plan.schema.is_none() {
        bail!(
            "{} was routed as a structured step but carries no schema",
            plan.step
        );
    }

    let resp = run_step(engines, registry, plan, system, user).await?;
    let text = resp.text().trim();

    if text.is_empty() {
        bail!(
            "{} returned empty output under guided decoding; \
             check that the deployment has a guided-decoding backend enabled",
            plan.step
        );
    }

    let parsed = serde_json::from_str::<T>(text).with_context(|| {
        format!(
            "{} returned output that does not match its schema: {}",
            plan.step,
            // Truncate: a runaway generation should not put kilobytes into a
            // log line, and the first part is where the problem is.
            text.chars().take(400).collect::<String>()
        )
    })?;
    Ok((parsed, resp))
}

/// Usage from a response, defaulted to zeros when the deployment omits it.
pub fn usage_of(resp: &ChatResponse) -> (u64, u64, u64) {
    match &resp.usage {
        Some(u) => (u.prompt_tokens, u.completion_tokens, u.cached_tokens()),
        None => (0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_output_parses_the_schemas_shape() {
        let json = r#"{"facts":[
            {"subject":"task:a","relation":"depends_on","object":"file:x","confidence":0.9},
            {"subject":"task:b","relation":"implements","object":"spec:y","confidence":0.4,"evidence":"stated"}
        ]}"#;
        let out: ExtractOutput = serde_json::from_str(json).unwrap();
        assert_eq!(out.facts.len(), 2);
        assert_eq!(out.facts[0].subject, "task:a");
        assert_eq!(out.facts[1].evidence.as_deref(), Some("stated"));
    }

    #[test]
    fn an_absent_facts_array_parses_as_empty() {
        let out: ExtractOutput = serde_json::from_str("{}").unwrap();
        assert!(out.facts.is_empty());
    }

    #[test]
    fn plan_output_parses() {
        let json = r#"{"steps":[{"intent":"read the file","tool":"read"}],"needs_tools":true}"#;
        let out: PlanOutput = serde_json::from_str(json).unwrap();
        assert_eq!(out.steps.len(), 1);
        assert_eq!(out.steps[0].tool.as_deref(), Some("read"));
        assert!(out.needs_tools);
    }

    #[test]
    fn route_output_parses_a_suggested_mood() {
        let json = r#"{"entry_points":["task:a"],"suggested_mood":"researcher"}"#;
        let out: RouteOutput = serde_json::from_str(json).unwrap();
        assert_eq!(out.entry_points, vec!["task:a".to_string()]);
        assert_eq!(out.suggested_mood.as_deref(), Some("researcher"));
    }

    #[test]
    fn usage_defaults_to_zeros_when_the_deployment_omits_it() {
        let resp = ChatResponse {
            id: String::new(),
            model: "m".into(),
            choices: vec![],
            usage: None,
        };
        assert_eq!(usage_of(&resp), (0, 0, 0));
    }
}
