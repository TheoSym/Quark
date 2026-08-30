//! The graph itself: storage, indices, conflict resolution, decay, promotion.

use qgi2_spec_types::{
    ConflictPolicy, Fact, FactId, FactKey, Relation, Source, TurnIndex,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Which slice of memory a fact belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Learned this session. Rendered into segment 5 (`subgraph`), which is
    /// volatile.
    Session,
    /// Promoted from previous sessions. Rendered into segment 3 (`durable`),
    /// which is part of the byte-stable cached prefix — so it must not change
    /// mid-session.
    Durable,
}

/// What happened when a fact was committed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommitOutcome {
    /// A triple the graph had not seen.
    Inserted(FactId),
    /// The same triple already existed; its confidence and reinforcement count
    /// were updated rather than a duplicate being created.
    Reinforced(FactId),
    /// A conflicting fact was replaced under `latest wins`.
    Superseded { new: FactId, old: FactId },
    /// A conflicting fact was kept alongside this one under `keep both`.
    CoexistsWith(FactId),
    /// The incoming fact lost a `highest confidence` comparison and was
    /// dropped.
    Rejected { reason: String },
}

impl CommitOutcome {
    pub fn id(&self) -> Option<&FactId> {
        match self {
            Self::Inserted(id) | Self::Reinforced(id) | Self::CoexistsWith(id) => Some(id),
            Self::Superseded { new, .. } => Some(new),
            Self::Rejected { .. } => None,
        }
    }

    pub fn changed_graph(&self) -> bool {
        !matches!(self, Self::Rejected { .. })
    }
}

/// An in-RAM typed fact graph.
///
/// Every collection is ordered. See the module docs for why that is a
/// correctness requirement rather than a preference.
/// Only `facts` and `scopes` are serialized. The three indices are derived
/// state, rebuilt by [`FactGraph::reindex`] on load: persisting them would
/// triple the snapshot size and let a hand-edited file disagree with itself.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactGraph {
    facts: BTreeMap<FactId, Fact>,
    scopes: BTreeMap<FactId, Scope>,
    /// subject -> fact ids
    #[serde(skip)]
    by_subject: BTreeMap<String, BTreeSet<FactId>>,
    /// object -> fact ids, for reverse traversal
    #[serde(skip)]
    by_object: BTreeMap<String, BTreeSet<FactId>>,
    /// (subject, relation) -> fact ids, the lookup conflict resolution needs
    #[serde(skip)]
    by_subject_relation: BTreeMap<(String, String), BTreeSet<FactId>>,
}

impl FactGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn get(&self, id: &FactId) -> Option<&Fact> {
        self.facts.get(id)
    }

    pub fn scope_of(&self, id: &FactId) -> Option<Scope> {
        self.scopes.get(id).copied()
    }

    /// Every live fact, in deterministic id order.
    pub fn iter_live(&self) -> impl Iterator<Item = &Fact> {
        self.facts.values().filter(|f| f.is_live())
    }

    /// Live facts in one scope, in deterministic id order.
    pub fn iter_scope(&self, scope: Scope) -> impl Iterator<Item = &Fact> {
        self.facts
            .values()
            .filter(move |f| f.is_live() && self.scopes.get(&f.id) == Some(&scope))
    }

    /// Live facts whose subject matches, in deterministic id order.
    pub fn by_subject(&self, subject: &str) -> impl Iterator<Item = &Fact> {
        self.by_subject
            .get(subject)
            .into_iter()
            .flatten()
            .filter_map(|id| self.facts.get(id))
            .filter(|f| f.is_live())
    }

    /// Live facts whose object matches, for reverse traversal.
    pub fn by_object(&self, object: &str) -> impl Iterator<Item = &Fact> {
        self.by_object
            .get(object)
            .into_iter()
            .flatten()
            .filter_map(|id| self.facts.get(id))
            .filter(|f| f.is_live())
    }

    /// Live facts sharing a `(subject, relation)` pair. These are the
    /// candidates a new fact can conflict with.
    pub fn by_subject_relation(&self, subject: &str, relation: &Relation) -> Vec<&Fact> {
        self.by_subject_relation
            .get(&(subject.to_string(), relation.as_str().to_string()))
            .into_iter()
            .flatten()
            .filter_map(|id| self.facts.get(id))
            .filter(|f| f.is_live())
            .collect()
    }

    /// Commit a rule-validated fact under a mood's conflict policy.
    ///
    /// This is the only write path. `qgi2-rules` calls it after verify; nothing
    /// else should, which is why the argument is a [`Fact`] (only constructible
    /// with a `CommitToken`) rather than a `ProposedFact`.
    pub fn commit(&mut self, fact: Fact, scope: Scope, policy: ConflictPolicy) -> CommitOutcome {
        // Exact same triple: reinforce rather than duplicate. This is what
        // keeps the graph small as a session repeats itself.
        if let Some(existing) = self.facts.get_mut(&fact.id)
            && existing.is_live()
        {
            existing.reinforcements = existing.reinforcements.saturating_add(1);
            // Take the higher confidence: a re-extraction that is more certain
            // should raise the entry, but a less certain one should not lower
            // something already well established.
            if fact.confidence > existing.confidence {
                existing.confidence = fact.confidence;
            }
            existing.turn = fact.turn;
            return CommitOutcome::Reinforced(fact.id);
        }

        let conflicts = self.find_conflicts(&fact);
        if conflicts.is_empty() {
            let id = fact.id.clone();
            self.insert_unchecked(fact, scope);
            return CommitOutcome::Inserted(id);
        }

        match policy {
            ConflictPolicy::KeepBoth => {
                let other = conflicts[0].clone();
                let id = fact.id.clone();
                self.insert_unchecked(fact, scope);
                let _ = id;
                CommitOutcome::CoexistsWith(other)
            }
            ConflictPolicy::LatestWins => {
                // The newest fact wins; older conflicting facts are marked
                // superseded but kept, so a Traceable run can still explain
                // what the graph used to believe.
                let old = conflicts[0].clone();
                let new_id = fact.id.clone();
                self.insert_unchecked(fact, scope);
                for c in conflicts {
                    if let Some(f) = self.facts.get_mut(&c) {
                        f.superseded_by = Some(new_id.clone());
                    }
                }
                CommitOutcome::Superseded { new: new_id, old }
            }
            ConflictPolicy::HighestConfidence => {
                let best = conflicts
                    .iter()
                    .filter_map(|id| self.facts.get(id))
                    .max_by(|a, b| {
                        a.confidence
                            .cmp(&b.confidence)
                            .then(a.reinforcements.cmp(&b.reinforcements))
                    })
                    .expect("conflicts is non-empty and every id resolves");

                let incoming_wins = fact.confidence > best.confidence;
                if incoming_wins {
                    let old = best.id.clone();
                    let new_id = fact.id.clone();
                    self.insert_unchecked(fact, scope);
                    for c in conflicts {
                        if let Some(f) = self.facts.get_mut(&c) {
                            f.superseded_by = Some(new_id.clone());
                        }
                    }
                    CommitOutcome::Superseded { new: new_id, old }
                } else {
                    CommitOutcome::Rejected {
                        reason: format!(
                            "confidence {:.2} does not beat existing {:.2} for {}",
                            fact.confidence.get(),
                            best.confidence.get(),
                            fact.key
                        ),
                    }
                }
            }
        }
    }

    /// Facts that contradict `fact`.
    ///
    /// Two kinds count as conflict: the same `(subject, relation)` pointing at
    /// a different object, and a `(subject, negated_relation, object)` that
    /// asserts the opposite. The second is why [`Relation::negation`] exists:
    /// `prefers`/`dislikes` on the same pair is a real contradiction that a
    /// same-relation check alone would miss.
    fn find_conflicts(&self, fact: &Fact) -> Vec<FactId> {
        let mut out = BTreeSet::new();

        for other in self.by_subject_relation(fact.subject(), fact.relation()) {
            if other.object() != fact.object() && other.id != fact.id {
                out.insert(other.id.clone());
            }
        }

        if let Some(neg) = fact.relation().negation() {
            for other in self.by_subject_relation(fact.subject(), &neg) {
                if other.object() == fact.object() {
                    out.insert(other.id.clone());
                }
            }
        }

        out.into_iter().collect()
    }

    fn insert_unchecked(&mut self, fact: Fact, scope: Scope) {
        let id = fact.id.clone();
        self.by_subject
            .entry(fact.key.subject.clone())
            .or_default()
            .insert(id.clone());
        self.by_object
            .entry(fact.key.object.clone())
            .or_default()
            .insert(id.clone());
        self.by_subject_relation
            .entry((
                fact.key.subject.clone(),
                fact.key.relation.as_str().to_string(),
            ))
            .or_default()
            .insert(id.clone());
        self.scopes.insert(id.clone(), scope);
        self.facts.insert(id, fact);
    }

    /// Remove a fact entirely, including from every index.
    ///
    /// Used by consolidation, not by the turn loop: the turn loop supersedes
    /// rather than deletes, so history survives.
    pub fn remove(&mut self, id: &FactId) -> Option<Fact> {
        let fact = self.facts.remove(id)?;
        self.scopes.remove(id);
        if let Some(set) = self.by_subject.get_mut(&fact.key.subject) {
            set.remove(id);
            if set.is_empty() {
                self.by_subject.remove(&fact.key.subject);
            }
        }
        if let Some(set) = self.by_object.get_mut(&fact.key.object) {
            set.remove(id);
            if set.is_empty() {
                self.by_object.remove(&fact.key.object);
            }
        }
        let sr = (
            fact.key.subject.clone(),
            fact.key.relation.as_str().to_string(),
        );
        if let Some(set) = self.by_subject_relation.get_mut(&sr) {
            set.remove(id);
            if set.is_empty() {
                self.by_subject_relation.remove(&sr);
            }
        }
        Some(fact)
    }

    /// Session-end decay.
    ///
    /// Facts lose confidence unless they were reinforced, and facts that fall
    /// under `floor` are dropped. This is what keeps "tokens per turn trending
    /// down as memory replaces raw context" from turning into an ever-growing
    /// durable slice.
    pub fn decay(&mut self, factor: f32, floor: f32) -> Vec<FactId> {
        let mut dropped = Vec::new();
        for fact in self.facts.values_mut() {
            // Repeatedly confirmed facts decay more slowly: dividing the decay
            // by the reinforcement count means something seen five times this
            // session survives a quiet session afterwards.
            let scaled = 1.0 - (1.0 - factor) / (fact.reinforcements as f32).max(1.0);
            fact.confidence = fact.confidence.decayed(scaled);
            if fact.confidence.get() < floor {
                dropped.push(fact.id.clone());
            }
        }
        for id in &dropped {
            self.remove(id);
        }
        dropped
    }

    /// Promote session facts into the durable slice at session end.
    ///
    /// Only facts at or above `min_confidence` and reinforced at least
    /// `min_reinforcements` times are promoted: the durable slice sits in the
    /// byte-stable cached prefix, so anything admitted there costs a cache
    /// rebuild at the start of every future session.
    pub fn promote_to_durable(
        &mut self,
        min_confidence: f32,
        min_reinforcements: u32,
    ) -> Vec<FactId> {
        let promoted: Vec<FactId> = self
            .facts
            .values()
            .filter(|f| {
                f.is_live()
                    && self.scopes.get(&f.id) == Some(&Scope::Session)
                    && f.confidence.get() >= min_confidence
                    && f.reinforcements >= min_reinforcements
            })
            .map(|f| f.id.clone())
            .collect();
        for id in &promoted {
            self.scopes.insert(id.clone(), Scope::Durable);
        }
        promoted
    }

    /// Drop every session-scoped fact, keeping the durable slice. Used when a
    /// session ends and its subgraph should not leak into the next one.
    pub fn clear_session(&mut self) {
        let ids: Vec<FactId> = self
            .facts
            .values()
            .filter(|f| self.scopes.get(&f.id) == Some(&Scope::Session))
            .map(|f| f.id.clone())
            .collect();
        for id in ids {
            self.remove(&id);
        }
    }

    /// Facts the harness wrote about itself (cache hit rates, acceptance
    /// rates), which the self-tuning loop reads back.
    pub fn harness_facts(&self) -> impl Iterator<Item = &Fact> {
        self.iter_live().filter(|f| f.source == Source::Harness)
    }

    /// All distinct subjects, in order. Used to seed lexical retrieval.
    pub fn subjects(&self) -> impl Iterator<Item = &str> {
        self.by_subject.keys().map(|s| s.as_str())
    }

    /// Serialize the graph. Field order is stable because every map is a
    /// `BTreeMap`, so a snapshot diff shows real changes rather than reordering.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut g: Self = serde_json::from_str(s)?;
        g.reindex();
        Ok(g)
    }

    /// Rebuild the derived indices from `facts`.
    ///
    /// Called after deserialization, where the indices arrive empty.
    pub fn reindex(&mut self) {
        self.by_subject.clear();
        self.by_object.clear();
        self.by_subject_relation.clear();
        for (id, fact) in &self.facts {
            self.by_subject
                .entry(fact.key.subject.clone())
                .or_default()
                .insert(id.clone());
            self.by_object
                .entry(fact.key.object.clone())
                .or_default()
                .insert(id.clone());
            self.by_subject_relation
                .entry((
                    fact.key.subject.clone(),
                    fact.key.relation.as_str().to_string(),
                ))
                .or_default()
                .insert(id.clone());
        }
    }

    /// Look up by triple without needing to build a [`Fact`].
    pub fn contains_key(&self, key: &FactKey) -> bool {
        self.facts.get(&key.id()).is_some_and(|f| f.is_live())
    }

    /// The highest turn index in the graph, for turn-based decay weighting.
    pub fn latest_turn(&self) -> TurnIndex {
        self.facts.values().map(|f| f.turn).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::{CommitToken, Confidence, ProposedFact};

    fn fact(subject: &str, rel: Relation, object: &str, conf: f32, turn: TurnIndex) -> Fact {
        ProposedFact {
            subject: subject.into(),
            relation: rel,
            object: object.into(),
            confidence: Confidence::new(conf),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, turn)
    }

    #[test]
    fn the_same_triple_reinforces_instead_of_duplicating() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.5, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let out = g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 2),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        assert!(matches!(out, CommitOutcome::Reinforced(_)));
        assert_eq!(g.len(), 1);
        let f = g.iter_live().next().unwrap();
        assert_eq!(f.reinforcements, 2);
        assert_eq!(f.confidence.get(), 0.9, "higher confidence should win");
    }

    #[test]
    fn reinforcing_never_lowers_confidence() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.2, 2),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        assert_eq!(g.iter_live().next().unwrap().confidence.get(), 0.9);
    }

    #[test]
    fn latest_wins_supersedes_the_older_fact() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let out = g.commit(
            fact("task:a", Relation::DependsOn, "file:y", 0.3, 2),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        assert!(matches!(out, CommitOutcome::Superseded { .. }));
        let live: Vec<_> = g.iter_live().map(|f| f.object().to_string()).collect();
        assert_eq!(live, vec!["file:y"], "lower confidence but newer still wins");
    }

    #[test]
    fn keep_both_retains_contradictory_claims() {
        let mut g = FactGraph::new();
        g.commit(
            fact("claim:c", Relation::Supports, "src:a", 0.8, 1),
            Scope::Session,
            ConflictPolicy::KeepBoth,
        );
        let out = g.commit(
            fact("claim:c", Relation::Contradicts, "src:a", 0.7, 2),
            Scope::Session,
            ConflictPolicy::KeepBoth,
        );
        assert!(matches!(out, CommitOutcome::CoexistsWith(_)));
        assert_eq!(g.iter_live().count(), 2);
    }

    #[test]
    fn highest_confidence_rejects_the_weaker_incoming_fact() {
        let mut g = FactGraph::new();
        g.commit(
            fact("person:p", Relation::Prefers, "topic:rust", 0.9, 1),
            Scope::Session,
            ConflictPolicy::HighestConfidence,
        );
        let out = g.commit(
            fact("person:p", Relation::Prefers, "topic:go", 0.4, 2),
            Scope::Session,
            ConflictPolicy::HighestConfidence,
        );
        assert!(matches!(out, CommitOutcome::Rejected { .. }));
        assert!(!out.changed_graph());
        assert_eq!(g.iter_live().count(), 1);
    }

    #[test]
    fn negated_relations_count_as_conflict() {
        // prefers(p, rust) and dislikes(p, rust) is a contradiction that a
        // same-relation check alone would miss.
        let mut g = FactGraph::new();
        g.commit(
            fact("person:p", Relation::Prefers, "topic:rust", 0.6, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let out = g.commit(
            fact("person:p", Relation::Dislikes, "topic:rust", 0.8, 2),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        assert!(matches!(out, CommitOutcome::Superseded { .. }));
        assert_eq!(g.iter_live().count(), 1);
    }

    #[test]
    fn decay_protects_reinforced_facts() {
        let mut g = FactGraph::new();
        // Seen once.
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.5, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        // Seen five times.
        for t in 1..=5 {
            g.commit(
                fact("task:b", Relation::DependsOn, "file:y", 0.5, t),
                Scope::Session,
                ConflictPolicy::LatestWins,
            );
        }
        g.decay(0.5, 0.0);
        let a = g.by_subject("task:a").next().unwrap().confidence.get();
        let b = g.by_subject("task:b").next().unwrap().confidence.get();
        assert!(b > a, "reinforced fact {b} should decay less than {a}");
    }

    #[test]
    fn decay_drops_facts_under_the_floor() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.2, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let dropped = g.decay(0.1, 0.1);
        assert_eq!(dropped.len(), 1);
        assert!(g.is_empty());
    }

    #[test]
    fn promotion_requires_both_confidence_and_reinforcement() {
        let mut g = FactGraph::new();
        // High confidence, seen once.
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.95, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        // Lower confidence, seen twice.
        for t in 1..=2 {
            g.commit(
                fact("task:b", Relation::DependsOn, "file:y", 0.8, t),
                Scope::Session,
                ConflictPolicy::LatestWins,
            );
        }
        let promoted = g.promote_to_durable(0.7, 2);
        assert_eq!(promoted.len(), 1);
        assert_eq!(g.iter_scope(Scope::Durable).count(), 1);
        assert_eq!(g.iter_scope(Scope::Durable).next().unwrap().subject(), "task:b");
    }

    #[test]
    fn clearing_the_session_keeps_the_durable_slice() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 1),
            Scope::Durable,
            ConflictPolicy::LatestWins,
        );
        g.commit(
            fact("task:b", Relation::DependsOn, "file:y", 0.9, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        g.clear_session();
        assert_eq!(g.len(), 1);
        assert_eq!(g.iter_scope(Scope::Durable).count(), 1);
    }

    #[test]
    fn removal_cleans_every_index() {
        let mut g = FactGraph::new();
        let f = fact("task:a", Relation::DependsOn, "file:x", 0.9, 1);
        let id = f.id.clone();
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        g.remove(&id);
        assert_eq!(g.by_subject("task:a").count(), 0);
        assert_eq!(g.by_object("file:x").count(), 0);
        assert_eq!(g.by_subject_relation("task:a", &Relation::DependsOn).len(), 0);
        assert!(g.subjects().next().is_none(), "empty index sets are pruned");
    }

    #[test]
    fn json_round_trips() {
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let json = g.to_json().unwrap();
        let back = FactGraph::from_json(&json).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.to_json().unwrap(), json);
    }

    #[test]
    fn indices_are_usable_after_a_round_trip() {
        // The indices are `serde(skip)`, so a load that forgot to reindex would
        // deserialize cleanly and then silently find nothing.
        let mut g = FactGraph::new();
        g.commit(
            fact("task:a", Relation::DependsOn, "file:x", 0.9, 1),
            Scope::Session,
            ConflictPolicy::LatestWins,
        );
        let back = FactGraph::from_json(&g.to_json().unwrap()).unwrap();
        assert_eq!(back.by_subject("task:a").count(), 1);
        assert_eq!(back.by_object("file:x").count(), 1);
        assert_eq!(
            back.by_subject_relation("task:a", &Relation::DependsOn).len(),
            1
        );
    }
}
