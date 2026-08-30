//! Deterministic rendering of the graph into prompt segments.
//!
//! Spec invariant:
//!
//! > Rendering is deterministic: same graph → same bytes → same cache blocks.
//!
//! Two consequences that shape this module:
//!
//! 1. **Nothing time-dependent, ever.** No timestamps, no "3 turns ago", no
//!    elapsed durations. A relative-time string would change the bytes on every
//!    turn for an unchanged graph and quietly destroy the cache.
//! 2. **A total order that does not depend on floats.** Facts sort by
//!    `(subject, relation, object)` — their identity — not by confidence.
//!    Sorting by confidence would reorder the whole segment when one fact's
//!    score moved by 0.01.

use crate::store::{FactGraph, Scope};
use qgi2_spec_types::{Fact, FactId};

/// How much of the graph may be rendered.
///
/// The spec wants the volatile tail short; a budget is how that is enforced
/// when the graph grows past what is useful to send.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderBudget {
    /// Maximum facts to render.
    pub max_facts: usize,
    /// Approximate byte ceiling for the rendered segment.
    pub max_bytes: usize,
}

impl Default for RenderBudget {
    fn default() -> Self {
        Self {
            max_facts: 128,
            max_bytes: 8 * 1024,
        }
    }
}

impl RenderBudget {
    pub const fn unlimited() -> Self {
        Self {
            max_facts: usize::MAX,
            max_bytes: usize::MAX,
        }
    }
}

/// A rendered segment plus what it had to leave out.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedSubgraph {
    pub text: String,
    pub rendered: usize,
    /// Facts dropped because the budget ran out. Reported so a chronically
    /// truncated subgraph is visible rather than silently degrading recall.
    pub truncated: usize,
}

impl RenderedSubgraph {
    pub fn is_empty(&self) -> bool {
        self.rendered == 0
    }
}

/// Render an explicit list of facts (typically a traversal result).
///
/// Facts are sorted by identity before rendering, so the caller's ordering —
/// which for a traversal is depth-first-then-id and therefore *does* depend on
/// entry points — cannot leak into the bytes. Two turns that retrieve the same
/// set of facts by different routes produce identical text.
pub fn render_facts(graph: &FactGraph, ids: &[FactId], budget: RenderBudget) -> RenderedSubgraph {
    let mut facts: Vec<&Fact> = ids
        .iter()
        .filter_map(|id| graph.get(id))
        .filter(|f| f.is_live())
        .collect();

    facts.sort_by(|a, b| a.key.cmp(&b.key));
    facts.dedup_by(|a, b| a.id == b.id);

    render_sorted(&facts, budget)
}

/// Render one scope of the graph in full — used for the durable slice, which is
/// segment 3 and therefore part of the byte-stable cached prefix.
pub fn render_scope(graph: &FactGraph, scope: Scope, budget: RenderBudget) -> RenderedSubgraph {
    let mut facts: Vec<&Fact> = graph.iter_scope(scope).collect();
    facts.sort_by(|a, b| a.key.cmp(&b.key));
    render_sorted(&facts, budget)
}

fn render_sorted(facts: &[&Fact], budget: RenderBudget) -> RenderedSubgraph {
    let mut text = String::new();
    let mut rendered = 0usize;

    for fact in facts {
        if rendered >= budget.max_facts {
            break;
        }
        let line = fact.render();
        // +1 for the newline this line will contribute.
        if !text.is_empty() && text.len() + line.len() + 1 > budget.max_bytes {
            break;
        }
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&line);
        rendered += 1;
    }

    RenderedSubgraph {
        text,
        rendered,
        truncated: facts.len().saturating_sub(rendered),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::{
        CommitToken, Confidence, ConflictPolicy, ProposedFact, Relation, Source,
    };

    fn add(g: &mut FactGraph, s: &str, r: Relation, o: &str, c: f32) -> FactId {
        let f = ProposedFact {
            subject: s.into(),
            relation: r,
            object: o.into(),
            confidence: Confidence::new(c),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
        let id = f.id.clone();
        g.commit(f, Scope::Session, ConflictPolicy::KeepBoth);
        id
    }

    #[test]
    fn render_order_does_not_depend_on_caller_order() {
        let mut g = FactGraph::new();
        let a = add(&mut g, "task:a", Relation::DependsOn, "file:x", 0.9);
        let b = add(&mut g, "task:b", Relation::DependsOn, "file:y", 0.5);
        let forward = render_facts(&g, &[a.clone(), b.clone()], RenderBudget::unlimited());
        let reverse = render_facts(&g, &[b, a], RenderBudget::unlimited());
        assert_eq!(forward.text, reverse.text);
    }

    #[test]
    fn render_order_does_not_depend_on_confidence() {
        // Sorting by score would reshuffle the whole segment when one fact's
        // confidence moved slightly, invalidating the cache for no reason.
        let mut g = FactGraph::new();
        add(&mut g, "task:a", Relation::DependsOn, "file:x", 0.1);
        add(&mut g, "task:b", Relation::DependsOn, "file:y", 0.99);
        let first = render_scope(&g, Scope::Session, RenderBudget::unlimited());
        assert!(first.text.starts_with("task:a"), "got {}", first.text);
    }

    #[test]
    fn rendering_is_byte_identical_across_calls() {
        let mut g = FactGraph::new();
        add(&mut g, "task:a", Relation::DependsOn, "file:x", 0.9);
        add(&mut g, "claim:c", Relation::Supports, "src:s", 0.4);
        let a = render_scope(&g, Scope::Session, RenderBudget::unlimited());
        let b = render_scope(&g, Scope::Session, RenderBudget::unlimited());
        assert_eq!(a, b);
    }

    #[test]
    fn superseded_facts_are_not_rendered() {
        let mut g = FactGraph::new();
        let old = add(&mut g, "task:a", Relation::DependsOn, "file:x", 0.9);
        let f = ProposedFact {
            subject: "task:a".into(),
            relation: Relation::DependsOn,
            object: "file:y".into(),
            confidence: Confidence::new(0.9),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 2);
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);

        let out = render_facts(&g, &[old], RenderBudget::unlimited());
        assert!(out.is_empty());
        let scope = render_scope(&g, Scope::Session, RenderBudget::unlimited());
        assert!(!scope.text.contains("file:x"));
    }

    #[test]
    fn budget_truncates_and_reports() {
        let mut g = FactGraph::new();
        for i in 0..10 {
            add(&mut g, &format!("task:{i}"), Relation::DependsOn, "file:x", 0.5);
        }
        let out = render_scope(
            &g,
            Scope::Session,
            RenderBudget {
                max_facts: 3,
                max_bytes: usize::MAX,
            },
        );
        assert_eq!(out.rendered, 3);
        assert_eq!(out.truncated, 7);
    }

    #[test]
    fn byte_budget_never_produces_a_partial_line() {
        let mut g = FactGraph::new();
        for i in 0..10 {
            add(&mut g, &format!("task:{i}"), Relation::DependsOn, "file:x", 0.5);
        }
        let out = render_scope(
            &g,
            Scope::Session,
            RenderBudget {
                max_facts: usize::MAX,
                max_bytes: 60,
            },
        );
        assert!(out.text.len() <= 60, "len {}", out.text.len());
        for line in out.text.lines() {
            assert!(line.ends_with(']'), "partial line: {line:?}");
        }
    }

    #[test]
    fn rendered_text_contains_nothing_time_dependent() {
        let mut g = FactGraph::new();
        add(&mut g, "task:a", Relation::DependsOn, "file:x", 0.9);
        let text = render_scope(&g, Scope::Session, RenderBudget::unlimited()).text;
        assert_eq!(text, "task:a depends_on file:x [0.90]");
    }
}
