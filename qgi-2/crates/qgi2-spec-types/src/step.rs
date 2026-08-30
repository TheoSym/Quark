//! Steps and the explicit `(model, speculation, sampling)` triple.
//!
//! Spec invariant:
//!
//! > Every step has an explicit `(model, speculation, sampling)` triple chosen
//! > by the router from mood and profile. Nothing defaults.
//!
//! "Nothing defaults" is why [`StepPlan`] has no `Default` impl and every field
//! is required: there is no way to construct one without stating all three
//! choices. The router is the only thing that builds them.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Which of the two models runs a step.
///
/// Spec: "Two models, one control layer. A big planner and a small worker,
/// each with its own speculation and cache, selected per step by the harness."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    /// Qwen3.8-Flash-Next (NVFP4). Plans and answers.
    Planner,
    /// Qwen3.8-27B (NVFP4). Extracts, renders, routes, produces tool args.
    Worker,
}

impl ModelRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planner => "planner",
            Self::Worker => "worker",
        }
    }
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A speculative decoding method and its lookahead.
///
/// Spec: "Speculate everywhere. MTP on the planner, DFlash2 or MTP on the
/// worker, n-gram where the output copies the prompt."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "method")]
pub enum Speculation {
    /// Multi-token prediction head. Supports greedy decoding.
    Mtp { n: u8 },
    /// DFlash2 draft model. Faster, but cannot produce greedy output, so it is
    /// unusable under the Deterministic profile.
    DFlash2 { n: u8 },
    /// N-gram lookup against the prompt. Only useful when output copies input.
    NGram { n: u8 },
    /// No speculation.
    Off,
}

impl Speculation {
    /// Whether this method can produce greedy (`T = 0`) output.
    ///
    /// The one place the DFlash2 limitation is encoded. Everything that pairs
    /// sampling with speculation asks here rather than matching on the variant.
    pub const fn supports_greedy(self) -> bool {
        match self {
            Self::Mtp { .. } | Self::NGram { .. } | Self::Off => true,
            Self::DFlash2 { .. } => false,
        }
    }

    /// Lookahead depth; `0` when speculation is off.
    pub const fn lookahead(self) -> u8 {
        match self {
            Self::Mtp { n } | Self::DFlash2 { n } | Self::NGram { n } => n,
            Self::Off => 0,
        }
    }

    pub const fn method_name(self) -> &'static str {
        match self {
            Self::Mtp { .. } => "mtp",
            Self::DFlash2 { .. } => "dflash2",
            Self::NGram { .. } => "ngram",
            Self::Off => "off",
        }
    }

    /// The acceptance floor this method is held to, from the spec's success
    /// metrics: planner MTP >= 1.8 tokens/step, worker DFlash2 >= 2.0.
    pub const fn acceptance_floor(self, role: ModelRole) -> Option<f64> {
        match (self, role) {
            (Self::Mtp { .. }, ModelRole::Planner) => Some(1.8),
            (Self::DFlash2 { .. }, ModelRole::Worker) => Some(2.0),
            _ => None,
        }
    }
}

impl fmt::Display for Speculation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            other => write!(f, "{} n={}", other.method_name(), other.lookahead()),
        }
    }
}

/// Sampling parameters for one step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    /// Fixed seed, set by the Deterministic profile.
    pub seed: Option<u64>,
    /// Request batch-invariant kernels so results do not depend on how
    /// requests were batched. Required for reproducibility under Deterministic.
    pub batch_invariant: bool,
    /// Whether extended thinking is enabled. Quick turns it off.
    pub thinking: bool,
    pub max_tokens: Option<u32>,
}

impl Sampling {
    /// The mood-table constructor: a temperature, everything else neutral.
    pub const fn at_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            top_p: 1.0,
            seed: None,
            batch_invariant: false,
            thinking: true,
            max_tokens: None,
        }
    }

    pub const fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// The stages of the per-turn loop.
///
/// Spec:
/// ```text
/// assemble → plan → tool calls → extract → verify → commit
///          → answer → extract answer facts → verify → commit → mood check
/// ```
/// `Verify` and `Commit` are rule stages rather than model calls, so they carry
/// no triple; [`StepKind::is_model_step`] distinguishes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Choose the model/speculation/sampling for the steps that follow.
    Route,
    /// Planner produces the plan.
    Plan,
    /// Worker fills in tool arguments under a schema, masked by the rules.
    ToolArgs,
    /// Worker extracts candidate facts under a schema.
    Extract,
    /// Rules: dedupe, conflict, confidence floor. Not a model call.
    Verify,
    /// Graph write and derived-view refresh. Not a model call.
    Commit,
    /// Planner produces the user-facing answer.
    Answer,
    /// Rules decide whether the mood should change. Not a model call.
    MoodCheck,
}

impl StepKind {
    pub const ALL: [StepKind; 8] = [
        StepKind::Route,
        StepKind::Plan,
        StepKind::ToolArgs,
        StepKind::Extract,
        StepKind::Verify,
        StepKind::Commit,
        StepKind::Answer,
        StepKind::MoodCheck,
    ];

    /// Whether this step calls a model (and therefore needs a triple).
    pub const fn is_model_step(self) -> bool {
        !matches!(self, Self::Verify | Self::Commit | Self::MoodCheck)
    }

    /// Whether this step must run under a JSON schema.
    ///
    /// Spec: "Every structured step (extract, render, tool-args, route) runs
    /// under a JSON schema. No free-text bookkeeping." `Plan` is included
    /// because a plan is bookkeeping the harness consumes, not prose for the
    /// user; `Answer` is the only free-text model step.
    pub const fn is_structured(self) -> bool {
        match self {
            Self::Route | Self::Plan | Self::ToolArgs | Self::Extract => true,
            Self::Answer | Self::Verify | Self::Commit | Self::MoodCheck => false,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Route => "route",
            Self::Plan => "plan",
            Self::ToolArgs => "tool_args",
            Self::Extract => "extract",
            Self::Verify => "verify",
            Self::Commit => "commit",
            Self::Answer => "answer",
            Self::MoodCheck => "mood_check",
        }
    }
}

impl fmt::Display for StepKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The explicit triple for one model step, plus its schema.
///
/// No `Default`: the spec says nothing defaults, so every plan states all three
/// choices. Only `qgi2-router` constructs these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepPlan {
    pub step: StepKind,
    pub role: ModelRole,
    pub speculation: Speculation,
    pub sampling: Sampling,
    /// JSON schema for structured steps, enforced by the engine's guided
    /// decoding. `None` only for [`StepKind::Answer`].
    pub schema: Option<serde_json::Value>,
}

impl StepPlan {
    /// Check the plan against the spec's invariants.
    ///
    /// Returns an error rather than panicking so the router can surface a
    /// misconfigured table as a startup failure.
    pub fn validate(&self) -> Result<(), String> {
        if !self.step.is_model_step() {
            return Err(format!("{} is a rule stage and takes no triple", self.step));
        }
        if self.sampling.is_greedy() && !self.speculation.supports_greedy() {
            return Err(format!(
                "{} pairs greedy sampling with {}, which cannot produce greedy output",
                self.step, self.speculation
            ));
        }
        if self.step.is_structured() && self.schema.is_none() {
            return Err(format!(
                "{} is a structured step and must run under a JSON schema",
                self.step
            ));
        }
        if !self.step.is_structured() && self.schema.is_some() {
            return Err(format!("{} is free-text and must not carry a schema", self.step));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_dflash2_refuses_greedy() {
        assert!(!Speculation::DFlash2 { n: 7 }.supports_greedy());
        assert!(Speculation::Mtp { n: 3 }.supports_greedy());
        assert!(Speculation::NGram { n: 4 }.supports_greedy());
        assert!(Speculation::Off.supports_greedy());
    }

    #[test]
    fn acceptance_floors_match_the_success_metrics() {
        assert_eq!(
            Speculation::Mtp { n: 2 }.acceptance_floor(ModelRole::Planner),
            Some(1.8)
        );
        assert_eq!(
            Speculation::DFlash2 { n: 7 }.acceptance_floor(ModelRole::Worker),
            Some(2.0)
        );
    }

    #[test]
    fn rule_stages_are_not_model_steps() {
        for s in [StepKind::Verify, StepKind::Commit, StepKind::MoodCheck] {
            assert!(!s.is_model_step());
        }
    }

    #[test]
    fn answer_is_the_only_unstructured_model_step() {
        let unstructured: Vec<_> = StepKind::ALL
            .into_iter()
            .filter(|s| s.is_model_step() && !s.is_structured())
            .collect();
        assert_eq!(unstructured, vec![StepKind::Answer]);
    }

    fn plan(step: StepKind, spec: Speculation, sampling: Sampling) -> StepPlan {
        StepPlan {
            step,
            role: ModelRole::Worker,
            speculation: spec,
            sampling,
            schema: step.is_structured().then(|| serde_json::json!({"type": "object"})),
        }
    }

    #[test]
    fn validate_rejects_greedy_with_dflash2() {
        let p = plan(
            StepKind::Extract,
            Speculation::DFlash2 { n: 7 },
            Sampling { temperature: 0.0, ..Sampling::at_temperature(0.0) },
        );
        assert!(p.validate().unwrap_err().contains("cannot produce greedy"));
    }

    #[test]
    fn validate_rejects_a_structured_step_without_a_schema() {
        let mut p = plan(
            StepKind::Extract,
            Speculation::Mtp { n: 3 },
            Sampling::at_temperature(0.7),
        );
        p.schema = None;
        assert!(p.validate().unwrap_err().contains("JSON schema"));
    }

    #[test]
    fn validate_rejects_a_schema_on_the_answer_step() {
        let mut p = plan(
            StepKind::Answer,
            Speculation::Mtp { n: 2 },
            Sampling::at_temperature(0.7),
        );
        p.schema = Some(serde_json::json!({"type": "object"}));
        assert!(p.validate().unwrap_err().contains("free-text"));
    }

    #[test]
    fn a_well_formed_plan_validates() {
        let p = plan(
            StepKind::Extract,
            Speculation::DFlash2 { n: 7 },
            Sampling::at_temperature(0.7),
        );
        assert!(p.validate().is_ok());
    }
}
