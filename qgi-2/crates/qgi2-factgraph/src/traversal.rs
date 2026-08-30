//! Mood-directed traversal of the fact graph.
//!
//! Each mood declares a chain in the spec's table — `Task→depends_on→File`,
//! `Claim→supports/contradicts→Source`, `Person→prefers→Topic`. Traversal walks
//! only the relations that mood follows, so a Builder turn never drags a
//! Companion's preference facts into the prompt.
//!
//! The walk is breadth-first with a depth cap from the profile
//! ([`RetrievalPolicy::max_depth`]) and visits neighbours in [`FactId`] order,
//! so the same graph and the same entry points always produce the same
//! subgraph in the same order.

use crate::store::FactGraph;
use qgi2_spec_types::{Fact, FactId, RetrievalPolicy, TraversalSpec};
use std::collections::{BTreeSet, VecDeque};

/// One breadth-first walk over the graph.
pub struct Walk<'g> {
    graph: &'g FactGraph,
    spec: &'g TraversalSpec,
    policy: RetrievalPolicy,
}

/// The facts a walk reached, with the depth each was found at.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TraversalResult {
    /// Reached facts in visit order: depth first, then [`FactId`] order within
    /// a depth. Stable for a given graph and entry set.
    pub facts: Vec<FactId>,
    /// Depth at which each fact was first reached, parallel to `facts`.
    pub depths: Vec<u8>,
    /// Entry points that matched nothing. Surfaced rather than swallowed so a
    /// retrieval that found nothing is visible instead of looking like an empty
    /// graph.
    pub unmatched_entries: Vec<String>,
}

impl TraversalResult {
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&FactId, u8)> {
        self.facts.iter().zip(self.depths.iter().copied())
    }
}

impl<'g> Walk<'g> {
    pub fn new(graph: &'g FactGraph, spec: &'g TraversalSpec, policy: RetrievalPolicy) -> Self {
        Self {
            graph,
            spec,
            policy,
        }
    }

    /// Walk outward from `entries` (subject strings), following only the
    /// mood's relations, up to the profile's depth cap.
    ///
    /// Traversal follows edges in both directions: a `Claim supports Source`
    /// fact should be reachable whether the entry point matched the claim or
    /// the source. Restricting to the forward direction would make retrieval
    /// depend on which end of the edge the embedding happened to hit.
    pub fn from_entries(&self, entries: &[String]) -> TraversalResult {
        let mut result = TraversalResult::default();
        let mut seen: BTreeSet<FactId> = BTreeSet::new();
        let mut seen_nodes: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<(String, u8)> = VecDeque::new();

        for entry in entries {
            if self.graph.by_subject(entry).next().is_none()
                && self.graph.by_object(entry).next().is_none()
            {
                result.unmatched_entries.push(entry.clone());
                continue;
            }
            if seen_nodes.insert(entry.clone()) {
                queue.push_back((entry.clone(), 0));
            }
        }

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= self.policy.max_depth {
                continue;
            }

            // Collect this node's edges in FactId order so the visit sequence
            // does not depend on index iteration details.
            let mut edges: BTreeSet<FactId> = BTreeSet::new();
            for f in self.graph.by_subject(&node) {
                if self.spec.follows(f.relation()) {
                    edges.insert(f.id.clone());
                }
            }
            for f in self.graph.by_object(&node) {
                if self.spec.follows(f.relation()) {
                    edges.insert(f.id.clone());
                }
            }

            for id in edges {
                let Some(fact) = self.graph.get(&id) else {
                    continue;
                };
                if seen.insert(id.clone()) {
                    result.facts.push(id.clone());
                    result.depths.push(depth);
                }
                // Enqueue the far end of the edge.
                for next in [fact.subject(), fact.object()] {
                    if next != node && seen_nodes.insert(next.to_string()) {
                        queue.push_back((next.to_string(), depth + 1));
                    }
                }
            }

            // `full_chain = false` (the Quick profile) stops after the entry
            // points' own edges rather than continuing along the chain.
            if !self.policy.full_chain {
                queue.clear();
            }
        }

        result
    }

    /// Resolve a traversal result back to facts, in visit order.
    pub fn resolve(&self, result: &TraversalResult) -> Vec<&'g Fact> {
        result
            .facts
            .iter()
            .filter_map(|id| self.graph.get(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Scope;
    use qgi2_spec_types::{
        CommitToken, Confidence, ConflictPolicy, Mood, Profile, ProposedFact, Relation, Source,
    };

    fn add(g: &mut FactGraph, s: &str, r: Relation, o: &str) {
        let f = ProposedFact {
            subject: s.into(),
            relation: r,
            object: o.into(),
            confidence: Confidence::new(0.9),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
        g.commit(f, Scope::Session, ConflictPolicy::KeepBoth);
    }

    fn builder_chain() -> FactGraph {
        let mut g = FactGraph::new();
        add(&mut g, "task:auth", Relation::DependsOn, "file:auth.rs");
        add(&mut g, "file:auth.rs", Relation::DependsOn, "file:db.rs");
        add(&mut g, "file:db.rs", Relation::DependsOn, "file:pool.rs");
        // A Companion-mood fact that Builder traversal must not follow.
        add(&mut g, "task:auth", Relation::Prefers, "topic:security");
        g
    }

    #[test]
    fn traversal_follows_only_the_moods_relations() {
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let r = walk.from_entries(&["task:auth".to_string()]);
        let objects: Vec<_> = walk
            .resolve(&r)
            .iter()
            .map(|f| f.object().to_string())
            .collect();
        assert!(!objects.contains(&"topic:security".to_string()));
        assert!(objects.contains(&"file:auth.rs".to_string()));
    }

    #[test]
    fn full_chain_reaches_the_far_end() {
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let r = walk.from_entries(&["task:auth".to_string()]);
        let objects: Vec<_> = walk
            .resolve(&r)
            .iter()
            .map(|f| f.object().to_string())
            .collect();
        assert!(objects.contains(&"file:pool.rs".to_string()), "got {objects:?}");
    }

    #[test]
    fn quick_profile_stops_at_depth_one() {
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Quick.retrieval());
        let r = walk.from_entries(&["task:auth".to_string()]);
        let objects: Vec<_> = walk
            .resolve(&r)
            .iter()
            .map(|f| f.object().to_string())
            .collect();
        assert!(objects.contains(&"file:auth.rs".to_string()));
        assert!(!objects.contains(&"file:pool.rs".to_string()), "got {objects:?}");
    }

    #[test]
    fn traversal_is_bidirectional() {
        // Entering at the object end must still find the edge; otherwise
        // retrieval depends on which end the embedder happened to hit.
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let r = walk.from_entries(&["file:pool.rs".to_string()]);
        assert!(!r.is_empty());
    }

    #[test]
    fn the_same_graph_and_entries_produce_the_same_order() {
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let a = walk.from_entries(&["task:auth".to_string()]);
        let b = walk.from_entries(&["task:auth".to_string()]);
        assert_eq!(a, b);
    }

    #[test]
    fn unmatched_entries_are_reported_not_swallowed() {
        let g = builder_chain();
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let r = walk.from_entries(&["task:nonexistent".to_string()]);
        assert_eq!(r.unmatched_entries, vec!["task:nonexistent".to_string()]);
        assert!(r.is_empty());
    }

    #[test]
    fn cycles_terminate() {
        let mut g = FactGraph::new();
        add(&mut g, "a", Relation::DependsOn, "b");
        add(&mut g, "b", Relation::DependsOn, "a");
        let spec = Mood::Builder.table().traversal;
        let walk = Walk::new(&g, &spec, Profile::Traceable.retrieval());
        let r = walk.from_entries(&["a".to_string()]);
        assert_eq!(r.len(), 2);
    }
}
