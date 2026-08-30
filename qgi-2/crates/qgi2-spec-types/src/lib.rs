//! Types for the QGI-2 inference-first harness.
//!
//! Every construct in `HARNESS_SPEC.md` that other crates must agree on lives
//! here: the fixed prompt segment order, the typed fact shape, the
//! `(model, speculation, sampling)` triple, and the mood/profile tables.
//!
//! This crate deliberately has no dependency on jcode. The jcode-facing code
//! lives in `qgi2-edge-provider`; keeping the spec types independent means the
//! control layer can be tested, and the HTTP edge can be served, without
//! linking the harness.

pub mod fact;
pub mod mood;
pub mod profile;
pub mod segment;
pub mod step;

pub use fact::{
    CommitToken, Confidence, Fact, FactId, FactKey, ProposedFact, Relation, Source, TurnIndex,
};
pub use mood::{ConflictPolicy, Mood, MoodTable, ToolClass, TraversalSpec};
pub use profile::{LoggingPolicy, MemorySync, Profile, ProfileTable, RetrievalPolicy};
pub use segment::{SEGMENT_ORDER, Segment, SegmentHash, SegmentId, SegmentSet};
pub use step::{ModelRole, Sampling, Speculation, StepKind, StepPlan};

use serde::{Deserialize, Serialize};

/// Errors produced by the QGI-2 control layer.
#[derive(Debug, thiserror::Error)]
pub enum Qgi2Error {
    /// A structured step returned output that did not satisfy its schema.
    ///
    /// The spec's "every structured step runs under a JSON schema" invariant
    /// means this is a hard failure, not something to paper over with a retry
    /// on free text.
    #[error("schema violation in {step:?}: {detail}")]
    Schema { step: StepKind, detail: String },

    /// The router was asked for a step it has no entry for. The spec requires
    /// every step to have an explicit triple ("Nothing defaults"), so a missing
    /// entry is a bug in the tables rather than a cue to pick something.
    #[error("no routing entry for step {step:?} under mood {mood:?} / profile {profile:?}")]
    UnroutedStep {
        step: StepKind,
        mood: Mood,
        profile: Profile,
    },

    /// A fact proposed by the model failed rule validation.
    #[error("fact rejected at verify: {reason}")]
    FactRejected { reason: String },

    /// The engine (vLLM) refused or failed a request.
    #[error("engine error: {0}")]
    Engine(String),

    /// Prompt assembly could not produce byte-stable output.
    #[error("assembly error: {0}")]
    Assembly(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Result alias for the control layer.
pub type Result<T> = std::result::Result<T, Qgi2Error>;

/// The pair that selects behaviour for a session.
///
/// Moods and profiles are orthogonal in the spec: a mood decides *what the
/// agent is doing* (which relations to traverse, which tools exist, how
/// conflicts resolve), a profile decides *how carefully it does it* (sampling
/// determinism, speculation, retrieval depth, logging).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Persona {
    pub mood: Mood,
    pub profile: Profile,
}

impl Persona {
    pub const fn new(mood: Mood, profile: Profile) -> Self {
        Self { mood, profile }
    }
}

impl Default for Persona {
    fn default() -> Self {
        Self {
            mood: Mood::Builder,
            profile: Profile::Traceable,
        }
    }
}

/// Success thresholds from the spec's "Success metrics" section.
///
/// These are stored rather than hardcoded at the check sites so the self-tuning
/// loop can read the same numbers it is trying to move.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    /// Prefix-cache hit rate per model. Spec: >= 0.85.
    pub cache_hit_rate: f64,
    /// Planner MTP acceptance, tokens per step. Spec: >= 1.8.
    pub planner_acceptance: f64,
    /// Worker DFlash2 acceptance, tokens per step. Spec: >= 2.0.
    pub worker_acceptance: f64,
    /// Fraction of extracted facts rejected at verify. Spec: <= 0.10.
    pub max_rejection_rate: f64,
    /// Planner:worker token ratio, expressed as planner/worker. Spec: <= 1/3.
    pub max_planner_worker_ratio: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            cache_hit_rate: 0.85,
            planner_acceptance: 1.8,
            worker_acceptance: 2.0,
            max_rejection_rate: 0.10,
            max_planner_worker_ratio: 1.0 / 3.0,
        }
    }
}
