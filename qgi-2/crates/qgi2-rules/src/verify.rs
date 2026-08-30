//! The verify stage: dedupe, conflict, confidence floor.
//!
//! Spec, per-turn loop:
//!
//! ```text
//! → extract     [worker, DFlash2 n=7 | MTP n=3, schema]
//! → verify      [rules: dedupe, conflict, confidence floor]
//! → commit      [graph write, derived views refresh]
//! ```
//!
//! This is the only place a [`CommitToken`] is minted, which is what makes
//! "the graph is never written by the model directly" true rather than
//! aspirational.
//!
//! The success metric this stage is held to is *"extraction rejection rate at
//! verify ≤ 10%"*. A high rejection rate is a signal about the extract step's
//! prompt or schema, not a reason to loosen the rules here — so
//! [`VerifyOutcome::rejection_rate`] is reported rather than acted on.

use crate::{scale, unscale};
use qgi2_factgraph::FactGraph;
use qgi2_spec_types::{
    CommitToken, ConflictPolicy, Fact, Mood, ProposedFact, Relation, Source, TurnIndex,
};
use serde::{Deserialize, Serialize};

/// Thresholds the verify rules apply.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerifyConfig {
    /// Facts below this confidence are rejected outright.
    pub confidence_floor: f32,
    /// Reject facts whose relation is outside the current mood's traversal set.
    ///
    /// On by default: a Builder turn that starts recording `prefers` facts is
    /// almost always the extractor drifting, and letting it through pollutes
    /// the graph that every later turn renders from.
    pub enforce_mood_relations: bool,
    /// Maximum subject/object length. Extractors that echo a whole paragraph
    /// into a subject produce facts that are never retrievable again.
    pub max_term_len: usize,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            confidence_floor: 0.35,
            enforce_mood_relations: true,
            max_term_len: 128,
        }
    }
}

/// Why a proposal did not make it into the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason", content = "detail")]
pub enum RejectionReason {
    /// Confidence below [`VerifyConfig::confidence_floor`].
    BelowFloor,
    /// Empty or over-long subject or object.
    Malformed(String),
    /// The same triple is already live in the graph at equal or higher
    /// confidence, so the commit would be a no-op reinforcement.
    Duplicate,
    /// Another proposal in the same batch asserted the same
    /// `(subject, relation)` with more confidence.
    LostToSiblingProposal(String),
    /// The relation is outside the current mood's traversal set.
    OutsideMood(String),
}

/// One rejected proposal, kept for the rejection-rate metric and for
/// Traceable-profile logging.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rejection {
    pub fact: ProposedFact,
    pub reason: RejectionReason,
}

/// What verify decided.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VerifyOutcome {
    /// Facts cleared for commit. These carry a [`CommitToken`] internally by
    /// construction.
    pub accepted: Vec<Fact>,
    pub rejected: Vec<Rejection>,
}

impl VerifyOutcome {
    pub fn total(&self) -> usize {
        self.accepted.len() + self.rejected.len()
    }

    /// Fraction of proposals rejected. Spec target: <= 0.10.
    ///
    /// Returns 0.0 for an empty batch rather than NaN, so a turn that extracted
    /// nothing does not poison the rolling average.
    pub fn rejection_rate(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        self.rejected.len() as f64 / self.total() as f64
    }
}

ascent::ascent! {
    /// Consistency rules over one batch of proposals plus the live graph.
    ///
    /// Indices into the caller's proposal vector are carried as `usize` so the
    /// program reasons about identity without cloning strings around.
    pub struct Consistency;

    // --- inputs ---

    /// (index, subject, relation, object, scaled confidence)
    relation proposed(usize, String, String, String, u32);
    /// (subject, relation, object, scaled confidence) already live in the graph
    relation existing(String, String, String, u32);
    /// (relation, its negation) — supplied for the relations that have one
    relation negates(String, String);
    /// Relations the current mood traverses
    relation mood_relation(String);
    /// Relations allowed regardless of mood (structural)
    relation structural(String);
    /// Scaled confidence floor
    relation floor(u32);
    /// Indices already rejected by the pre-pass (malformed input)
    relation malformed(usize);

    // --- derived ---

    /// Confidence strictly below the floor.
    relation below_floor(usize);
    below_floor(i) <-- proposed(i, _, _, _, c), floor(f), if c < f;

    /// The graph already holds this triple at least as confidently.
    relation duplicate(usize);
    duplicate(i) <--
        proposed(i, s, r, o, c),
        existing(s, r, o, ec),
        if ec >= c;

    /// The relation is neither in the mood's traversal set nor structural.
    relation outside_mood(usize);
    outside_mood(i) <--
        proposed(i, _, r, _, _),
        !mood_relation(r),
        !structural(r);

    /// Two proposals in this batch assert the same (subject, relation) with
    /// different objects: they cannot both be the answer.
    relation sibling_conflict(usize, usize);
    sibling_conflict(i, j) <--
        proposed(i, s, r, o1, _),
        proposed(j, s, r, o2, _),
        if i != j,
        if o1 != o2;

    // Two proposals in this batch assert a relation and its negation over the
    // same pair. (ascent only accepts doc comments on relation declarations,
    // so rule-level notes are line comments.)
    sibling_conflict(i, j) <--
        proposed(i, s, r1, o, _),
        proposed(j, s, r2, o, _),
        negates(r1, r2),
        if i != j;

    /// A proposal that lost a sibling conflict on confidence. Ties are broken
    /// by index so the outcome does not depend on batch iteration order.
    relation lost_sibling(usize, usize);
    lost_sibling(i, j) <--
        sibling_conflict(i, j),
        proposed(i, _, _, _, ci),
        proposed(j, _, _, _, cj),
        if (cj > ci) || (cj == ci && j < i);

    /// Survived every rule.
    relation accepted(usize);
    accepted(i) <--
        proposed(i, _, _, _, _),
        !below_floor(i),
        !duplicate(i),
        !outside_mood(i),
        !malformed(i),
        !lost_sibling(i, _);
}

/// Run the verify rules over a batch of proposals.
///
/// `mood` supplies the traversal relations and, through
/// [`ConflictPolicy::KeepBoth`], decides whether contradictory relations are
/// allowed to coexist — under Researcher, `supports` and `contradicts` on the
/// same pair is the point, not an error.
pub fn verify(
    proposals: Vec<ProposedFact>,
    graph: &FactGraph,
    mood: Mood,
    source: Source,
    turn: TurnIndex,
    config: VerifyConfig,
) -> VerifyOutcome {
    let table = mood.table();
    let mut prog = Consistency::default();

    // Pre-pass: structural well-formedness, which is cheaper to check in Rust
    // than to express as a rule and produces a better error message.
    let mut malformed_detail: Vec<Option<String>> = vec![None; proposals.len()];
    for (i, p) in proposals.iter().enumerate() {
        let bad = if p.subject.trim().is_empty() {
            Some("empty subject".to_string())
        } else if p.object.trim().is_empty() {
            Some("empty object".to_string())
        } else if p.subject.len() > config.max_term_len {
            Some(format!(
                "subject is {} bytes, over the {} limit",
                p.subject.len(),
                config.max_term_len
            ))
        } else if p.object.len() > config.max_term_len {
            Some(format!(
                "object is {} bytes, over the {} limit",
                p.object.len(),
                config.max_term_len
            ))
        } else {
            None
        };
        if bad.is_some() {
            prog.malformed.push((i,));
        }
        malformed_detail[i] = bad;

        prog.proposed.push((
            i,
            p.subject.clone(),
            p.relation.as_str().to_string(),
            p.object.clone(),
            scale(p.confidence.get()),
        ));
    }

    for f in graph.iter_live() {
        prog.existing.push((
            f.key.subject.clone(),
            f.key.relation.as_str().to_string(),
            f.key.object.clone(),
            scale(f.confidence.get()),
        ));
    }

    // KeepBoth is exactly the policy that says contradictory relations are
    // evidence rather than error, so the negation rules are not loaded for it.
    if table.conflict != ConflictPolicy::KeepBoth {
        for r in [
            Relation::Supports,
            Relation::Contradicts,
            Relation::Prefers,
            Relation::Dislikes,
        ] {
            if let Some(n) = r.negation() {
                prog.negates
                    .push((r.as_str().to_string(), n.as_str().to_string()));
            }
        }
    }

    if config.enforce_mood_relations {
        for r in &table.traversal.relations {
            prog.mood_relation.push((r.as_str().to_string(),));
        }
        for r in [Relation::IsA, Relation::PartOf] {
            prog.structural.push((r.as_str().to_string(),));
        }
    } else {
        // With enforcement off, every proposed relation counts as structural,
        // so `outside_mood` derives nothing.
        for p in &proposals {
            prog.structural.push((p.relation.as_str().to_string(),));
        }
    }

    prog.floor.push((scale(config.confidence_floor),));
    prog.run();

    let accepted_idx: std::collections::BTreeSet<usize> =
        prog.accepted.iter().map(|(i,)| *i).collect();
    let below: std::collections::BTreeSet<usize> =
        prog.below_floor.iter().map(|(i,)| *i).collect();
    let dup: std::collections::BTreeSet<usize> = prog.duplicate.iter().map(|(i,)| *i).collect();
    let outside: std::collections::BTreeSet<usize> =
        prog.outside_mood.iter().map(|(i,)| *i).collect();
    let lost: std::collections::BTreeMap<usize, usize> =
        prog.lost_sibling.iter().map(|(i, j)| (*i, *j)).collect();

    let mut outcome = VerifyOutcome::default();
    for (i, p) in proposals.into_iter().enumerate() {
        if accepted_idx.contains(&i) {
            outcome
                .accepted
                .push(p.commit(CommitToken::issued_by_verify_stage(), source.clone(), turn));
            continue;
        }

        // Report the most specific reason. Order matters: a malformed fact that
        // is also below the floor should read as malformed.
        let reason = if let Some(detail) = malformed_detail[i].clone() {
            RejectionReason::Malformed(detail)
        } else if below.contains(&i) {
            RejectionReason::BelowFloor
        } else if outside.contains(&i) {
            RejectionReason::OutsideMood(format!(
                "{} is not traversed by the {} mood",
                p.relation, mood
            ))
        } else if dup.contains(&i) {
            RejectionReason::Duplicate
        } else if let Some(winner) = lost.get(&i) {
            RejectionReason::LostToSiblingProposal(format!("proposal {winner} was more confident"))
        } else {
            // Unreachable given the rule set, but a silent drop here would be
            // invisible in the rejection-rate metric.
            RejectionReason::Malformed("rejected by an unnamed rule".into())
        };
        outcome.rejected.push(Rejection { fact: p, reason });
    }

    outcome
}

/// Confidence of a fact as the rules see it, for tests and diagnostics.
pub fn scaled_confidence(f: &Fact) -> f32 {
    unscale(scale(f.confidence.get()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_factgraph::Scope;
    use qgi2_spec_types::Confidence;

    fn prop(s: &str, r: Relation, o: &str, c: f32) -> ProposedFact {
        ProposedFact {
            subject: s.into(),
            relation: r,
            object: o.into(),
            confidence: Confidence::new(c),
            evidence: None,
        }
    }

    fn empty() -> FactGraph {
        FactGraph::new()
    }

    #[test]
    fn facts_below_the_floor_are_rejected() {
        let out = verify(
            vec![prop("task:a", Relation::DependsOn, "file:x", 0.1)],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert!(out.accepted.is_empty());
        assert_eq!(out.rejected[0].reason, RejectionReason::BelowFloor);
    }

    #[test]
    fn well_formed_facts_are_accepted() {
        let out = verify(
            vec![prop("task:a", Relation::DependsOn, "file:x", 0.9)],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1);
        assert_eq!(out.rejection_rate(), 0.0);
    }

    #[test]
    fn empty_terms_are_malformed() {
        let out = verify(
            vec![
                prop("", Relation::DependsOn, "file:x", 0.9),
                prop("task:a", Relation::DependsOn, "   ", 0.9),
            ],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 0);
        assert!(matches!(
            out.rejected[0].reason,
            RejectionReason::Malformed(_)
        ));
        assert!(matches!(
            out.rejected[1].reason,
            RejectionReason::Malformed(_)
        ));
    }

    #[test]
    fn over_long_terms_are_malformed() {
        let long = "x".repeat(500);
        let out = verify(
            vec![prop(&long, Relation::DependsOn, "file:x", 0.9)],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert!(matches!(
            out.rejected[0].reason,
            RejectionReason::Malformed(_)
        ));
    }

    #[test]
    fn a_triple_the_graph_already_holds_more_confidently_is_a_duplicate() {
        let mut g = empty();
        let f = prop("task:a", Relation::DependsOn, "file:x", 0.9).commit(
            CommitToken::issued_by_verify_stage(),
            Source::User,
            1,
        );
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);

        let out = verify(
            vec![prop("task:a", Relation::DependsOn, "file:x", 0.5)],
            &g,
            Mood::Builder,
            Source::User,
            2,
            VerifyConfig::default(),
        );
        assert_eq!(out.rejected[0].reason, RejectionReason::Duplicate);
    }

    #[test]
    fn a_more_confident_re_extraction_is_not_a_duplicate() {
        // It should reach commit, where it reinforces and raises confidence.
        let mut g = empty();
        let f = prop("task:a", Relation::DependsOn, "file:x", 0.5).commit(
            CommitToken::issued_by_verify_stage(),
            Source::User,
            1,
        );
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);

        let out = verify(
            vec![prop("task:a", Relation::DependsOn, "file:x", 0.9)],
            &g,
            Mood::Builder,
            Source::User,
            2,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1);
    }

    #[test]
    fn the_more_confident_of_two_sibling_proposals_wins() {
        let out = verify(
            vec![
                prop("task:a", Relation::DependsOn, "file:x", 0.6),
                prop("task:a", Relation::DependsOn, "file:y", 0.9),
            ],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1);
        assert_eq!(out.accepted[0].object(), "file:y");
        assert!(matches!(
            out.rejected[0].reason,
            RejectionReason::LostToSiblingProposal(_)
        ));
    }

    #[test]
    fn sibling_ties_break_deterministically_by_index() {
        // Equal confidence must not depend on iteration order, or the same
        // batch produces different graphs on different runs.
        let batch = vec![
            prop("task:a", Relation::DependsOn, "file:x", 0.8),
            prop("task:a", Relation::DependsOn, "file:y", 0.8),
        ];
        let a = verify(batch.clone(), &empty(), Mood::Builder, Source::User, 1, VerifyConfig::default());
        let b = verify(batch, &empty(), Mood::Builder, Source::User, 1, VerifyConfig::default());
        assert_eq!(a.accepted.len(), 1);
        assert_eq!(a.accepted[0].object(), b.accepted[0].object());
        assert_eq!(a.accepted[0].object(), "file:x", "lower index wins ties");
    }

    #[test]
    fn relations_outside_the_mood_are_rejected() {
        let out = verify(
            vec![prop("person:p", Relation::Prefers, "topic:rust", 0.9)],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert!(matches!(
            out.rejected[0].reason,
            RejectionReason::OutsideMood(_)
        ));
    }

    #[test]
    fn structural_relations_survive_any_mood() {
        for mood in Mood::ALL {
            let out = verify(
                vec![prop("file:x", Relation::PartOf, "module:auth", 0.9)],
                &empty(),
                mood,
                Source::User,
                1,
                VerifyConfig::default(),
            );
            assert_eq!(out.accepted.len(), 1, "{mood} rejected a structural relation");
        }
    }

    #[test]
    fn researcher_keeps_contradictory_claims_in_one_batch() {
        // KeepBoth is the whole point of the Researcher mood: supports and
        // contradicts over the same pair is evidence of a live disagreement.
        let out = verify(
            vec![
                prop("claim:c", Relation::Supports, "src:s", 0.8),
                prop("claim:c", Relation::Contradicts, "src:s", 0.7),
            ],
            &empty(),
            Mood::Researcher,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 2, "got {:?}", out.rejected);
    }

    #[test]
    fn companion_does_not_keep_both_sides_of_a_preference() {
        let out = verify(
            vec![
                prop("person:p", Relation::Prefers, "topic:rust", 0.9),
                prop("person:p", Relation::Dislikes, "topic:rust", 0.4),
            ],
            &empty(),
            Mood::Companion,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1);
        assert_eq!(*out.accepted[0].relation(), Relation::Prefers);
    }

    #[test]
    fn disabling_mood_enforcement_admits_any_relation() {
        let out = verify(
            vec![prop("person:p", Relation::Prefers, "topic:rust", 0.9)],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig {
                enforce_mood_relations: false,
                ..VerifyConfig::default()
            },
        );
        assert_eq!(out.accepted.len(), 1);
    }

    #[test]
    fn rejection_rate_is_zero_for_an_empty_batch() {
        let out = verify(
            vec![],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.rejection_rate(), 0.0);
        assert!(!out.rejection_rate().is_nan());
    }

    #[test]
    fn every_rejection_carries_a_named_reason() {
        // A rejection with no reason would be invisible in the metric and
        // impossible to debug from a Traceable log.
        let out = verify(
            vec![
                prop("", Relation::DependsOn, "file:x", 0.9),
                prop("task:a", Relation::DependsOn, "file:x", 0.01),
                prop("person:p", Relation::Prefers, "t", 0.9),
            ],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.rejected.len(), 3);
        for r in &out.rejected {
            assert!(
                !matches!(&r.reason, RejectionReason::Malformed(m) if m.contains("unnamed")),
                "unnamed rejection: {r:?}"
            );
        }
    }
}
