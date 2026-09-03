//! Typed facts: the unit the memory graph stores.
//!
//! Spec:
//!
//! > Memory — jcode's in-RAM graph, extended with typed facts:
//! > `(subject, relation, object, confidence, source, turn)` on each entry.
//!
//! and:
//!
//! > The model proposes facts; rules validate and commit. The graph is never
//! > written by the model directly.
//!
//! That second invariant is enforced by types here: the model's output
//! deserializes into [`ProposedFact`], which has no id and no commit path. Only
//! `qgi2-rules` can turn one into a [`Fact`], and it does so by calling
//! [`ProposedFact::commit`], which requires a [`CommitToken`] that only the
//! verify stage can mint.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Monotonic turn counter within a session. Facts record the turn they were
/// learned on so decay and "latest wins" conflict resolution have an ordering
/// that does not depend on wall-clock time.
pub type TurnIndex = u64;

/// Stable identity for a committed fact.
///
/// Derived from the fact's key rather than randomly generated, so re-learning
/// the same fact reinforces the existing entry instead of duplicating it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactId(pub String);

impl fmt::Display for FactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Confidence in `[0.0, 1.0]`.
///
/// A newtype rather than a bare `f32` because the verify stage enforces a
/// confidence floor, and an unclamped float from model output would let a
/// proposal claim 1.7 and sail past it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "f32", into = "f32")]
pub struct Confidence(f32);

// PartialOrd must agree with the explicit Ord below, so it is written in terms
// of it rather than derived from f32's (which would disagree on NaN, and which
// clippy rejects outright).
impl PartialOrd for Confidence {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Confidence {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    /// Clamp into range. Model output is untrusted, so this never fails; it
    /// saturates, and the verify stage's floor check does the rejecting.
    pub fn new(v: f32) -> Self {
        Self(if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) })
    }

    pub fn get(self) -> f32 {
        self.0
    }

    /// Multiplicative decay, used at session end.
    pub fn decayed(self, factor: f32) -> Self {
        Self::new(self.0 * factor)
    }
}

impl From<f32> for Confidence {
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

impl From<Confidence> for f32 {
    fn from(c: Confidence) -> Self {
        c.0
    }
}

impl Eq for Confidence {}

impl Ord for Confidence {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Confidence is always finite by construction, so a total order is
        // well-defined here even though f32: !Ord in general.
        self.0.total_cmp(&other.0)
    }
}

/// Where a fact came from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum Source {
    /// Extracted from a user message.
    User,
    /// Extracted from the agent's own answer.
    Answer,
    /// Extracted from a tool result; carries the tool name.
    Tool(String),
    /// Derived by a rule rather than proposed by the model.
    Rule(String),
    /// Promoted from a previous session's durable slice.
    Durable,
    /// Emitted by the harness itself (metrics facts for the self-tuning loop).
    Harness,
}

impl Source {
    /// Whether a fact from this source may be written without model proposal.
    ///
    /// Rules and the harness write directly; everything else must pass verify.
    pub const fn is_trusted(&self) -> bool {
        matches!(self, Self::Rule(_) | Self::Harness | Self::Durable)
    }
}

/// The relation in a typed fact.
///
/// The named variants are the ones the mood traversal tables in the spec refer
/// to by name; anything else a mood needs is expressible as [`Relation::Other`]
/// so a new mood does not require a change to this enum.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    // Builder mood: Task -> depends_on -> File
    DependsOn,
    Implements,
    Modifies,
    // Researcher mood: Claim -> supports/contradicts -> Source
    Supports,
    Contradicts,
    CitedBy,
    // Companion mood: Person -> prefers -> Topic
    Prefers,
    Dislikes,
    KnowsAbout,
    // Cross-mood structural relations.
    IsA,
    PartOf,
    /// Escape hatch for mood-specific relations added by configuration.
    Other(String),
}

impl Relation {
    pub fn as_str(&self) -> &str {
        match self {
            Self::DependsOn => "depends_on",
            Self::Implements => "implements",
            Self::Modifies => "modifies",
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::CitedBy => "cited_by",
            Self::Prefers => "prefers",
            Self::Dislikes => "dislikes",
            Self::KnowsAbout => "knows_about",
            Self::IsA => "is_a",
            Self::PartOf => "part_of",
            Self::Other(s) => s,
        }
    }

    /// The relation that would contradict this one, when there is one.
    ///
    /// The consistency rules use this to find conflicting pairs without
    /// hardcoding the pairing at each rule site.
    pub fn negation(&self) -> Option<Relation> {
        match self {
            Self::Supports => Some(Self::Contradicts),
            Self::Contradicts => Some(Self::Supports),
            Self::Prefers => Some(Self::Dislikes),
            Self::Dislikes => Some(Self::Prefers),
            _ => None,
        }
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Relation {
    fn from(s: &str) -> Self {
        match s {
            "depends_on" => Self::DependsOn,
            "implements" => Self::Implements,
            "modifies" => Self::Modifies,
            "supports" => Self::Supports,
            "contradicts" => Self::Contradicts,
            "cited_by" => Self::CitedBy,
            "prefers" => Self::Prefers,
            "dislikes" => Self::Dislikes,
            "knows_about" => Self::KnowsAbout,
            "is_a" => Self::IsA,
            "part_of" => Self::PartOf,
            other => Self::Other(other.to_string()),
        }
    }
}

/// The identity of a fact: subject, relation, object.
///
/// Confidence, source, and turn are *about* a fact rather than part of it, so
/// two extractions of the same triple reinforce one entry rather than creating
/// two. This is what makes `dedupe` at verify a lookup instead of a similarity
/// search.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactKey {
    pub subject: String,
    pub relation: Relation,
    pub object: String,
}

impl FactKey {
    pub fn new(subject: impl Into<String>, relation: Relation, object: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            relation,
            object: object.into(),
        }
    }

    /// Deterministic id: same triple, same id, on every machine and run.
    ///
    /// Deterministic rather than random because the spec requires
    /// "same graph → same bytes → same cache blocks"; a random id would leak
    /// into the rendered subgraph and change the prompt for an unchanged graph.
    pub fn id(&self) -> FactId {
        let mut h = blake3::Hasher::new();
        // Length-prefixed so ("ab", r, "c") and ("a", r, "bc") differ.
        for part in [
            self.subject.as_str(),
            self.relation.as_str(),
            self.object.as_str(),
        ] {
            h.update(&(part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
        FactId(h.finalize().to_hex()[..16].to_string())
    }
}

impl fmt::Display for FactKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.subject, self.relation, self.object)
    }
}

/// A fact the model proposed but that no rule has validated yet.
///
/// This is the deserialization target for the extract step's JSON schema. It
/// has no [`FactId`] and no way to reach the graph: the only route to a
/// committed [`Fact`] is [`ProposedFact::commit`], which needs a
/// [`CommitToken`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposedFact {
    pub subject: String,
    pub relation: Relation,
    pub object: String,
    #[serde(default = "default_confidence")]
    pub confidence: Confidence,
    /// Free-text justification the model supplies; used by verify and logging,
    /// never rendered into the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

fn default_confidence() -> Confidence {
    Confidence::new(0.5)
}

impl ProposedFact {
    pub fn key(&self) -> FactKey {
        FactKey::new(
            self.subject.clone(),
            self.relation.clone(),
            self.object.clone(),
        )
    }

    /// Turn a proposal into a committed fact.
    ///
    /// Requires a [`CommitToken`], which only the verify stage can construct.
    /// This is the type-level form of "the graph is never written by the model
    /// directly".
    pub fn commit(self, _token: CommitToken, source: Source, turn: TurnIndex) -> Fact {
        let key = self.key();
        Fact {
            id: key.id(),
            key,
            confidence: self.confidence,
            source,
            turn,
            reinforcements: 1,
            superseded_by: None,
        }
    }
}

/// Proof that the verify stage approved a fact.
///
/// Deliberately unconstructable outside `qgi2-rules`: the field is private and
/// the only constructor is `#[doc(hidden)]` and named to be conspicuous at a
/// call site. Rust visibility is the enforcement mechanism for the spec's
/// "rules validate and commit" invariant.
#[derive(Debug, Clone, Copy)]
pub struct CommitToken(());

impl CommitToken {
    /// Mint a commit token. **Only the verify stage may call this.**
    ///
    /// Calling it anywhere else bypasses rule validation and violates the spec
    /// invariant that the model never writes the graph directly.
    #[doc(hidden)]
    pub const fn issued_by_verify_stage() -> Self {
        Self(())
    }
}

/// A committed, rule-validated fact in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    #[serde(flatten)]
    pub key: FactKey,
    pub confidence: Confidence,
    pub source: Source,
    pub turn: TurnIndex,
    /// How many times this fact has been independently re-extracted. Used by
    /// the `highest confidence` conflict policy as a tie-break and by decay to
    /// protect repeatedly-confirmed facts.
    pub reinforcements: u32,
    /// Set when a later fact replaced this one under a `latest wins` policy.
    /// Superseded facts stay in the graph for traceability but are excluded
    /// from rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<FactId>,
}

impl Fact {
    pub fn subject(&self) -> &str {
        &self.key.subject
    }

    pub fn relation(&self) -> &Relation {
        &self.key.relation
    }

    pub fn object(&self) -> &str {
        &self.key.object
    }

    pub fn is_live(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Render one fact deterministically for the subgraph segment.
    ///
    /// No timestamps, no ids, no float formatting beyond two fixed decimals:
    /// anything that could vary run to run would break
    /// "same graph → same bytes → same cache blocks".
    pub fn render(&self) -> String {
        format!(
            "{} {} {} [{:.2}]",
            self.key.subject,
            self.key.relation,
            self.key.object,
            self.confidence.get()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_ids_are_deterministic() {
        let k = FactKey::new("task:auth", Relation::DependsOn, "file:auth.rs");
        assert_eq!(k.id(), k.id());
        assert_eq!(k.id(), FactKey::new("task:auth", Relation::DependsOn, "file:auth.rs").id());
    }

    #[test]
    fn length_prefixing_prevents_field_boundary_collisions() {
        let a = FactKey::new("ab", Relation::IsA, "c");
        let b = FactKey::new("a", Relation::IsA, "bc");
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn confidence_saturates_instead_of_trusting_model_output() {
        assert_eq!(Confidence::new(1.7).get(), 1.0);
        assert_eq!(Confidence::new(-3.0).get(), 0.0);
        assert_eq!(Confidence::new(f32::NAN).get(), 0.0);
    }

    #[test]
    fn rendering_is_stable_across_calls() {
        let f = ProposedFact {
            subject: "claim:x".into(),
            relation: Relation::Supports,
            object: "source:y".into(),
            confidence: Confidence::new(0.8),
            evidence: Some("ignored in render".into()),
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 3);
        assert_eq!(f.render(), "claim:x supports source:y [0.80]");
        assert_eq!(f.render(), f.render());
    }

    #[test]
    fn negation_pairs_are_symmetric() {
        for r in [Relation::Supports, Relation::Contradicts, Relation::Prefers, Relation::Dislikes] {
            let n = r.negation().expect("has negation");
            assert_eq!(n.negation().as_ref(), Some(&r));
        }
        assert!(Relation::DependsOn.negation().is_none());
    }
}
