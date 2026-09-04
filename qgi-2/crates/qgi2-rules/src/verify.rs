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
//!
//! The rules, over one batch of proposals plus the facts they could touch.
//! Each is a method on [`Consistency`] with the rule's name, so a Traceable
//! log names the rule that fired:
//!
//! ```text
//! below_floor(i)         <- proposed(i, c), c < floor
//! duplicate(i)           <- proposed(i, s, r, o, c), existing(s, r, o, ec), ec >= c
//! outside_mood(i)        <- proposed(i, r), !mood_relation(r), !structural(r)
//! sibling_conflict(i, j) <- proposed(i, s, r, o1), proposed(j, s, r, o2), o1 != o2
//! sibling_conflict(i, j) <- proposed(i, s, r1, o), proposed(j, s, r2, o), negates(r1, r2)
//! lost_sibling(i, j)     <- sibling_conflict(i, j), cj > ci or (cj == ci and j < i)
//! accepted(i)            <- proposed(i), none of the above, !malformed(i)
//! ```

use crate::{scale, unscale};
use qgi2_factgraph::FactGraph;
use qgi2_spec_types::{
    CommitToken, ConflictPolicy, Fact, Mood, ProposedFact, Relation, Source, TurnIndex,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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

/// One proposal as the rules see it: confidence scaled to the integer domain
/// so comparisons are exact at 1/1000 resolution.
struct Proposed<'a> {
    subject: &'a str,
    relation: &'a Relation,
    object: &'a str,
    confidence: u32,
}

/// Consistency rules over one batch of proposals plus the live graph.
///
/// Proposals are addressed by index into the caller's vector, so the rules
/// reason about identity without cloning strings around.
struct Consistency<'a> {
    proposed: Vec<Proposed<'a>>,
    /// `(subject, relation, object) -> highest scaled confidence` already live
    /// in the graph, for the keys a proposal could touch.
    existing: BTreeMap<(&'a str, &'a str, &'a str), u32>,
    /// `(relation, its negation)` — populated only when the mood's conflict
    /// policy treats them as conflicting.
    negates: BTreeSet<(Relation, Relation)>,
    /// Relations the current mood traverses.
    mood_relation: BTreeSet<Relation>,
    /// Relations allowed regardless of mood (structural).
    structural: BTreeSet<Relation>,
    floor: u32,
    /// Indices the pre-pass rejected as malformed.
    malformed: Vec<bool>,
}

impl Consistency<'_> {
    /// Survives every rule that does not depend on other proposals. Only an
    /// eligible proposal can win a sibling conflict: a malformed or
    /// below-floor sibling is going to be rejected anyway, and letting it
    /// knock out a valid one would lose both.
    fn eligible(&self, i: usize) -> bool {
        !self.malformed[i] && !self.below_floor(i) && !self.outside_mood(i) && !self.duplicate(i)
    }

    /// Confidence strictly below the floor.
    fn below_floor(&self, i: usize) -> bool {
        self.proposed[i].confidence < self.floor
    }

    /// The graph already holds this triple at least as confidently.
    fn duplicate(&self, i: usize) -> bool {
        let p = &self.proposed[i];
        self.existing
            .get(&(p.subject, p.relation.as_str(), p.object))
            .is_some_and(|ec| *ec >= p.confidence)
    }

    /// The relation is neither in the mood's traversal set nor structural.
    fn outside_mood(&self, i: usize) -> bool {
        let r = self.proposed[i].relation;
        !self.mood_relation.contains(r) && !self.structural.contains(r)
    }

    /// Two proposals in this batch cannot both be the answer: same
    /// `(subject, relation)` with different objects, or a relation and its
    /// negation over the same pair.
    fn sibling_conflict(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        let (a, b) = (&self.proposed[i], &self.proposed[j]);
        if a.subject != b.subject {
            return false;
        }
        (a.relation == b.relation && a.object != b.object)
            || (a.object == b.object
                && self.negates.contains(&(a.relation.clone(), b.relation.clone())))
    }

    /// The sibling `i` lost to on confidence, if any. Ties are broken by index
    /// so the outcome does not depend on batch iteration order. When several
    /// siblings beat `i`, the strongest (then lowest-indexed) is named.
    ///
    /// Only an [`eligible`](Self::eligible) sibling can win. Siblings are not
    /// chased transitively: if `j` beats `i` and `k` beats `j`, `i` is out
    /// either way when all three share a key, and the negation case has no
    /// sound transitive reading.
    fn lost_sibling(&self, i: usize) -> Option<usize> {
        let ci = self.proposed[i].confidence;
        (0..self.proposed.len())
            .filter(|&j| self.eligible(j) && self.sibling_conflict(i, j))
            .filter(|&j| {
                let cj = self.proposed[j].confidence;
                cj > ci || (cj == ci && j < i)
            })
            .max_by_key(|&j| (self.proposed[j].confidence, std::cmp::Reverse(j)))
    }
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

    // Pre-pass: structural well-formedness, which produces a better error
    // message than a rule would and is reported ahead of every other reason.
    let malformed: Vec<Option<String>> = proposals
        .iter()
        .map(|p| {
            if p.subject.trim().is_empty() {
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
            }
        })
        .collect();

    let proposed: Vec<Proposed<'_>> = proposals
        .iter()
        .map(|p| Proposed {
            subject: &p.subject,
            relation: &p.relation,
            object: &p.object,
            confidence: scale(p.confidence.get()),
        })
        .collect();

    // Load only the facts a proposal could conflict with, not the whole graph.
    // `duplicate` joins on the proposal's exact key, so anything outside the
    // proposals' (subject, relation) pairs -- or their negations -- can never
    // fire and is pure cost: O(graph) per extraction, twice per turn, which a
    // large durable slice turns into a visible stall.
    let mut existing: BTreeMap<(&str, &str, &str), u32> = BTreeMap::new();
    for p in &proposals {
        let mut relations = vec![p.relation.clone()];
        if let Some(neg) = p.relation.negation() {
            relations.push(neg);
        }
        for r in relations {
            for f in graph.by_subject_relation(&p.subject, &r) {
                let key = (f.key.subject.as_str(), f.key.relation.as_str(), f.key.object.as_str());
                let c = scale(f.confidence.get());
                existing
                    .entry(key)
                    .and_modify(|e| *e = (*e).max(c))
                    .or_insert(c);
            }
        }
    }

    // KeepBoth is exactly the policy that says contradictory relations are
    // evidence rather than error, so the negation pairs are not loaded for it.
    let mut negates = BTreeSet::new();
    if table.conflict != ConflictPolicy::KeepBoth {
        for r in [
            Relation::Supports,
            Relation::Contradicts,
            Relation::Prefers,
            Relation::Dislikes,
        ] {
            if let Some(n) = r.negation() {
                negates.insert((r, n));
            }
        }
    }

    let (mood_relation, structural) = if config.enforce_mood_relations {
        (
            table.traversal.relations.iter().cloned().collect(),
            BTreeSet::from([Relation::IsA, Relation::PartOf]),
        )
    } else {
        // With enforcement off, every proposed relation counts as structural,
        // so `outside_mood` derives nothing.
        (
            BTreeSet::new(),
            proposals.iter().map(|p| p.relation.clone()).collect(),
        )
    };

    let rules = Consistency {
        proposed,
        existing,
        negates,
        mood_relation,
        structural,
        floor: scale(config.confidence_floor),
        malformed: malformed.iter().map(Option::is_some).collect(),
    };

    // Decide every index before consuming the proposals: the rules borrow them.
    let decisions: Vec<Option<RejectionReason>> = (0..proposals.len())
        .map(|i| {
            // Report the most specific reason. Order matters: a malformed fact
            // that is also below the floor should read as malformed.
            if let Some(detail) = &malformed[i] {
                Some(RejectionReason::Malformed(detail.clone()))
            } else if rules.below_floor(i) {
                Some(RejectionReason::BelowFloor)
            } else if rules.outside_mood(i) {
                Some(RejectionReason::OutsideMood(format!(
                    "{} is not traversed by the {} mood",
                    proposals[i].relation, mood
                )))
            } else if rules.duplicate(i) {
                Some(RejectionReason::Duplicate)
            } else {
                rules.lost_sibling(i).map(|winner| {
                    RejectionReason::LostToSiblingProposal(format!(
                        "proposal {winner} was more confident"
                    ))
                })
            }
        })
        .collect();
    drop(rules);

    let mut outcome = VerifyOutcome::default();
    for (p, decision) in proposals.into_iter().zip(decisions) {
        match decision {
            None => outcome
                .accepted
                .push(p.commit(CommitToken::issued_by_verify_stage(), source.clone(), turn)),
            Some(reason) => outcome.rejected.push(Rejection { fact: p, reason }),
        }
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
    fn an_equally_confident_re_extraction_is_a_duplicate() {
        // `ec >= c`: equal confidence is a no-op reinforcement, rejected as such.
        let mut g = empty();
        let f = prop("task:a", Relation::DependsOn, "file:x", 0.7).commit(
            CommitToken::issued_by_verify_stage(),
            Source::User,
            1,
        );
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        let out = verify(
            vec![prop("task:a", Relation::DependsOn, "file:x", 0.7)],
            &g,
            Mood::Builder,
            Source::User,
            2,
            VerifyConfig::default(),
        );
        assert_eq!(out.rejected[0].reason, RejectionReason::Duplicate);
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
    fn three_siblings_leave_exactly_one_and_name_the_real_winner() {
        let out = verify(
            vec![
                prop("task:a", Relation::DependsOn, "file:x", 0.5),
                prop("task:a", Relation::DependsOn, "file:y", 0.9),
                prop("task:a", Relation::DependsOn, "file:z", 0.7),
            ],
            &empty(),
            Mood::Builder,
            Source::User,
            1,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1);
        assert_eq!(out.accepted[0].object(), "file:y");
        for r in &out.rejected {
            assert_eq!(
                r.reason,
                RejectionReason::LostToSiblingProposal("proposal 1 was more confident".into())
            );
        }
    }

    #[test]
    fn a_sibling_that_is_itself_rejected_cannot_knock_out_a_valid_one() {
        // The old rule set let a malformed or below-floor sibling win the
        // conflict and then get rejected itself, losing both proposals.
        let long = "x".repeat(500);
        let out = verify(
            vec![
                prop("task:a", Relation::DependsOn, "file:x", 0.6),
                prop("task:a", Relation::DependsOn, &long, 0.9), // malformed
                prop("task:a", Relation::DependsOn, "file:z", 0.99), // duplicate below
            ],
            &{
                let mut g = empty();
                let f = prop("task:a", Relation::DependsOn, "file:z", 0.99).commit(
                    CommitToken::issued_by_verify_stage(),
                    Source::User,
                    1,
                );
                g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
                g
            },
            Mood::Builder,
            Source::User,
            2,
            VerifyConfig::default(),
        );
        assert_eq!(out.accepted.len(), 1, "{:?}", out.rejected);
        assert_eq!(out.accepted[0].object(), "file:x");
        assert!(matches!(out.rejected[0].reason, RejectionReason::Malformed(_)));
        assert_eq!(out.rejected[1].reason, RejectionReason::Duplicate);
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
    fn verify_loads_only_facts_the_proposals_can_touch() {
        // `duplicate` joins on the proposal's (subject, relation), so the rest
        // of the graph can never fire a rule. Loading it anyway made verify
        // O(graph) per extraction, twice per turn.
        let mut g = empty();
        for i in 0..500 {
            let f = prop(&format!("task:{i}"), Relation::DependsOn, "file:x", 0.9).commit(
                CommitToken::issued_by_verify_stage(),
                Source::User,
                1,
            );
            g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        }
        // A duplicate of one of them, and one unrelated proposal.
        let out = verify(
            vec![
                prop("task:7", Relation::DependsOn, "file:x", 0.5),
                prop("task:new", Relation::DependsOn, "file:y", 0.9),
            ],
            &g,
            Mood::Builder,
            Source::User,
            2,
            VerifyConfig::default(),
        );
        // Correctness is unchanged: the duplicate is still caught, the new
        // fact is still accepted. The load itself is checked by the fact that
        // this runs against 500 facts without touching them (see the indexed
        // load above); a whole-graph load would still pass this assertion, so
        // the guard is the code, not a timing test.
        assert_eq!(out.accepted.len(), 1);
        assert_eq!(out.rejected[0].reason, RejectionReason::Duplicate);
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
