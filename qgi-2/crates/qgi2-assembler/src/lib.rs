//! Cache-shaped prompt assembly.
//!
//! Spec:
//!
//! > **Make the prompt cache-shaped.** Fixed segment order, stable bytes at the
//! > front, volatile bytes at the end. The KV/recurrent-state cache does the
//! > rest.
//!
//! and:
//!
//! > Cache hit rate is measured every turn from vLLM's `cached_tokens` and
//! > surfaced in the UI. **A drop below threshold is a bug, not a metric.**
//!
//! The assembler is where that becomes actionable. It remembers the previous
//! turn's segment hashes and reports, before the request is sent, whether the
//! stable prefix changed — so a cache miss is attributed to the segment that
//! caused it rather than discovered afterwards as an unexplained number.

mod core_prompt;

pub use core_prompt::CORE_PROMPT;

use qgi2_factgraph::{FactGraph, RenderBudget, Scope, render};
use qgi2_spec_types::{
    Mood, Profile, Segment, SegmentHash, SegmentId, SegmentSet, SEGMENT_ORDER,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What the assembler expects the cache to do with this prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheOutlook {
    /// First turn of a session; nothing is cached yet.
    ColdStart,
    /// The stable prefix is unchanged. Only the volatile tail should reprefill.
    PrefixIntact {
        /// Bytes expected to hit the cache.
        stable_bytes: usize,
        /// Bytes expected to reprefill.
        volatile_bytes: usize,
    },
    /// A byte-stable segment changed. The spec calls this a bug: segments 1–3
    /// must not move within a session.
    PrefixBroken {
        /// Which stable segments changed, in prompt order.
        changed: Vec<SegmentId>,
        /// The earliest changed segment — everything from here on reprefills.
        first_changed: SegmentId,
        explanation: String,
    },
}

impl CacheOutlook {
    pub fn is_broken(&self) -> bool {
        matches!(self, Self::PrefixBroken { .. })
    }

    /// Fraction of the prompt expected to hit the cache, for the pre-flight
    /// estimate. The authoritative number comes from vLLM's `cached_tokens`
    /// after the fact; this is what lets the harness flag a miss it *caused*
    /// versus one the engine surprised it with.
    pub fn expected_hit_ratio(&self) -> f64 {
        match self {
            Self::ColdStart | Self::PrefixBroken { .. } => 0.0,
            Self::PrefixIntact {
                stable_bytes,
                volatile_bytes,
            } => {
                let total = stable_bytes + volatile_bytes;
                if total == 0 {
                    0.0
                } else {
                    *stable_bytes as f64 / total as f64
                }
            }
        }
    }
}

/// One assembled prompt plus its cache bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct Assembled {
    pub segments: SegmentSet,
    pub outlook: CacheOutlook,
    /// Facts the subgraph segment had to drop for budget. Surfaced so a
    /// chronically truncated subgraph is visible.
    pub subgraph_truncated: usize,
}

impl Assembled {
    /// The cacheable system prompt: segments 1–3.
    pub fn system(&self) -> String {
        self.segments.render_stable()
    }

    /// The recomputed tail: segments 4–6.
    pub fn volatile(&self) -> String {
        self.segments.render_volatile()
    }

    /// Segment hashes for logging under the Traceable and Deterministic
    /// profiles.
    pub fn hash_log(&self) -> Vec<(String, String)> {
        self.segments
            .iter()
            .map(|s| (s.id.to_string(), s.hash.short()))
            .collect()
    }
}

/// Assembles prompts for one session and tracks its stable prefix.
///
/// One instance per session: the whole point is remembering what the previous
/// turn's segments hashed to.
#[derive(Debug, Clone, Default)]
pub struct Assembler {
    previous: Option<BTreeMap<SegmentId, SegmentHash>>,
    budget: RenderBudget,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_budget(budget: RenderBudget) -> Self {
        Self {
            previous: None,
            budget,
        }
    }

    /// Whether this assembler has seen a turn yet.
    pub fn is_cold(&self) -> bool {
        self.previous.is_none()
    }

    /// Forget the previous turn's hashes. Called on a deliberate prefix change
    /// — a mood switch, say — so the next turn reports `ColdStart` rather than
    /// `PrefixBroken` for a change the harness intended.
    pub fn reset(&mut self) {
        self.previous = None;
    }

    /// Assemble the six segments in spec order.
    ///
    /// `skills` and `query` are the caller's; `durable` and `subgraph` are
    /// rendered from the graph. Rendering happens here rather than at the call
    /// site so the determinism guarantees in [`qgi2_factgraph::render`] cannot
    /// be bypassed by a caller formatting facts its own way.
    pub fn assemble(
        &mut self,
        graph: &FactGraph,
        mood: Mood,
        _profile: Profile,
        skills: &[String],
        subgraph_ids: &[qgi2_spec_types::FactId],
        query: &str,
    ) -> Assembled {
        let durable = render::render_scope(graph, Scope::Durable, self.budget);
        let subgraph = render::render_facts(graph, subgraph_ids, self.budget);

        let segments = SegmentSet::new(
            CORE_PROMPT.to_string(),
            mood.table().render(),
            durable.text,
            render_skills(skills),
            subgraph.text,
            query.to_string(),
        );

        let outlook = self.check_cache(&segments);
        self.previous = Some(segments.hashes().into_iter().collect());

        Assembled {
            segments,
            outlook,
            subgraph_truncated: subgraph.truncated,
        }
    }

    fn check_cache(&self, segments: &SegmentSet) -> CacheOutlook {
        let Some(prev) = &self.previous else {
            return CacheOutlook::ColdStart;
        };

        let changed: Vec<SegmentId> = SEGMENT_ORDER
            .into_iter()
            .filter(|id| id.is_stable())
            .filter(|id| prev.get(id) != Some(&segments.get(*id).hash))
            .collect();

        if changed.is_empty() {
            let stable_bytes: usize = segments.stable_prefix().map(|s| s.text.len()).sum();
            let volatile_bytes: usize = segments.volatile_tail().map(|s| s.text.len()).sum();
            return CacheOutlook::PrefixIntact {
                stable_bytes,
                volatile_bytes,
            };
        }

        let first_changed = changed[0];
        CacheOutlook::PrefixBroken {
            explanation: format!(
                "segment {} ({}) changed mid-session; everything from position {} on will reprefill. \
                 Segments 1-3 are required to be byte-stable within a session.",
                first_changed.position(),
                first_changed,
                first_changed.position()
            ),
            first_changed,
            changed,
        }
    }
}

/// Render the active skill list into segment 4.
///
/// Sorted and deduplicated: the set is what matters, and an unstable order
/// would recompute the segment when nothing actually changed.
fn render_skills(skills: &[String]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut sorted: Vec<&String> = skills.iter().collect();
    sorted.sort();
    sorted.dedup();
    let mut out = String::from("# Active skills\n");
    for (i, s) in sorted.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str("- ");
        out.push_str(s);
    }
    out
}

/// Convenience accessor used by the edges.
pub fn segment_of(assembled: &Assembled, id: SegmentId) -> &Segment {
    assembled.segments.get(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::{
        CommitToken, Confidence, ConflictPolicy, ProposedFact, Relation, Source,
    };

    fn graph() -> FactGraph {
        let mut g = FactGraph::new();
        let f = ProposedFact {
            subject: "task:a".into(),
            relation: Relation::DependsOn,
            object: "file:x".into(),
            confidence: Confidence::new(0.9),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
        g.commit(f, Scope::Durable, ConflictPolicy::LatestWins);
        g
    }

    #[test]
    fn the_first_turn_is_a_cold_start() {
        let mut a = Assembler::new();
        let out = a.assemble(&graph(), Mood::Builder, Profile::Traceable, &[], &[], "hi");
        assert_eq!(out.outlook, CacheOutlook::ColdStart);
        assert_eq!(out.outlook.expected_hit_ratio(), 0.0);
    }

    #[test]
    fn changing_only_the_query_leaves_the_prefix_intact() {
        let g = graph();
        let mut a = Assembler::new();
        a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "first");
        let out = a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "second");
        assert!(matches!(out.outlook, CacheOutlook::PrefixIntact { .. }));
        assert!(out.outlook.expected_hit_ratio() > 0.0);
    }

    #[test]
    fn changing_skills_leaves_the_prefix_intact() {
        // Skills are segment 4 — volatile by design, so activating one must not
        // cost the cached prefix.
        let g = graph();
        let mut a = Assembler::new();
        a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        let out = a.assemble(
            &g,
            Mood::Builder,
            Profile::Traceable,
            &["rust-review".into()],
            &[],
            "q",
        );
        assert!(matches!(out.outlook, CacheOutlook::PrefixIntact { .. }), "{:?}", out.outlook);
    }

    #[test]
    fn switching_mood_breaks_the_prefix_and_says_which_segment() {
        let g = graph();
        let mut a = Assembler::new();
        a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        let out = a.assemble(&g, Mood::Researcher, Profile::Traceable, &[], &[], "q");
        match out.outlook {
            CacheOutlook::PrefixBroken {
                first_changed,
                ref changed,
                ref explanation,
            } => {
                assert_eq!(first_changed, SegmentId::Mood);
                assert_eq!(*changed, vec![SegmentId::Mood]);
                assert!(explanation.contains("byte-stable"));
            }
            other => panic!("expected PrefixBroken, got {other:?}"),
        }
        assert_eq!(out.outlook.expected_hit_ratio(), 0.0);
    }

    #[test]
    fn writing_to_the_durable_slice_mid_session_breaks_the_prefix() {
        // The durable slice is segment 3. Promoting facts into it mid-session
        // is exactly the mistake the outlook is meant to catch.
        let mut g = graph();
        let mut a = Assembler::new();
        a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");

        let f = ProposedFact {
            subject: "task:b".into(),
            relation: Relation::DependsOn,
            object: "file:y".into(),
            confidence: Confidence::new(0.9),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 2);
        g.commit(f, Scope::Durable, ConflictPolicy::LatestWins);

        let out = a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        match out.outlook {
            CacheOutlook::PrefixBroken { first_changed, .. } => {
                assert_eq!(first_changed, SegmentId::Durable);
            }
            other => panic!("expected PrefixBroken, got {other:?}"),
        }
    }

    #[test]
    fn reset_turns_an_intended_change_into_a_cold_start() {
        let g = graph();
        let mut a = Assembler::new();
        a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        a.reset();
        let out = a.assemble(&g, Mood::Researcher, Profile::Traceable, &[], &[], "q");
        assert_eq!(out.outlook, CacheOutlook::ColdStart, "not reported as a bug");
    }

    #[test]
    fn the_system_prompt_is_exactly_the_stable_prefix() {
        let mut a = Assembler::new();
        let out = a.assemble(&graph(), Mood::Builder, Profile::Traceable, &[], &[], "q");
        assert!(out.system().starts_with(CORE_PROMPT));
        assert!(out.system().contains("# Mood: builder"));
        assert!(!out.system().contains('q'.to_string().as_str()) || !out.system().ends_with("q"));
        assert!(out.volatile().ends_with("q"));
    }

    #[test]
    fn skill_order_does_not_affect_the_bytes() {
        let g = graph();
        let mut a = Assembler::new();
        let one = a.assemble(
            &g,
            Mood::Builder,
            Profile::Traceable,
            &["b".into(), "a".into()],
            &[],
            "q",
        );
        let mut b = Assembler::new();
        let two = b.assemble(
            &g,
            Mood::Builder,
            Profile::Traceable,
            &["a".into(), "b".into()],
            &[],
            "q",
        );
        assert_eq!(
            one.segments.get(SegmentId::Skills).hash,
            two.segments.get(SegmentId::Skills).hash
        );
    }

    #[test]
    fn assembly_is_deterministic_for_the_same_inputs() {
        let g = graph();
        let mut a = Assembler::new();
        let one = a.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        let mut b = Assembler::new();
        let two = b.assemble(&g, Mood::Builder, Profile::Traceable, &[], &[], "q");
        assert_eq!(one.segments, two.segments);
    }
}
