//! Tool gating.
//!
//! Spec: the mood table's `Tools` row — Builder gets `fs, shell, git`,
//! Researcher gets `web, fetch, docs`, Companion gets `calendar, mail, notes` —
//! and the per-turn loop's `tool calls [args under schema; mask from rules]`.
//!
//! The mask is applied on QGI-2's side of the seam: the tool list is filtered
//! *before* it reaches the model, so a masked tool is one the model never sees
//! and therefore never calls. jcode's own tool registry is untouched.
//!
//! Gating is more than a mood lookup because facts can deny a tool the mood
//! would otherwise allow — a `repo is_a read_only` fact takes write tools away
//! from Builder without changing the mood.

use qgi2_factgraph::FactGraph;
use qgi2_spec_types::{Mood, Relation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The tools a turn may use, and why the rest were removed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMask {
    /// Tool names the model will be shown, sorted.
    pub allowed: Vec<String>,
    /// Tools removed because the mood does not admit them.
    pub denied_by_mood: Vec<String>,
    /// Tools removed by a fact-driven rule, with the rule's name.
    pub denied_by_rule: Vec<(String, String)>,
}

impl ToolMask {
    pub fn permits(&self, tool: &str) -> bool {
        self.allowed.iter().any(|t| t == tool)
    }

    /// Reason a tool is unavailable, for the message the harness returns when
    /// the model asks for one anyway.
    pub fn denial_reason(&self, tool: &str) -> Option<String> {
        if let Some((_, rule)) = self.denied_by_rule.iter().find(|(t, _)| t == tool) {
            return Some(format!("denied by rule `{rule}`"));
        }
        if self.denied_by_mood.iter().any(|t| t == tool) {
            return Some("not admitted by the current mood".to_string());
        }
        None
    }
}

ascent::ascent! {
    /// Tool gating over the mood's tool classes and the graph's policy facts.
    pub struct Gating;

    /// A tool the harness offers.
    relation available(String);
    /// A tool name the current mood's classes cover.
    relation mood_admits(String);
    /// (tool, rule name) — a fact-driven denial.
    relation denied(String, String);

    /// Tools the model will be shown.
    relation allowed(String);
    allowed(t) <-- available(t), mood_admits(t), !denied(t, _);

    /// Available, mood-admitted, but denied by a rule.
    relation rule_blocked(String, String);
    rule_blocked(t, r) <-- available(t), mood_admits(t), denied(t, r);

    /// Available but outside the mood.
    relation mood_blocked(String);
    mood_blocked(t) <-- available(t), !mood_admits(t);
}

/// Tools that write to the filesystem or run commands. A read-only rule
/// removes exactly these.
const MUTATING_TOOLS: &[&str] = &["write", "edit", "multiedit", "patch", "apply_patch", "bash", "bg"];

/// Compute the tool mask for a turn.
///
/// `available` is the harness's full tool list (for the jcode edge, the tool
/// names jcode advertises). Facts consulted:
///
/// - `<anything> is_a read_only` — removes [`MUTATING_TOOLS`].
/// - `tool:<name> is_a forbidden` — removes that tool by name.
pub fn tool_mask(available: &[String], mood: Mood, graph: &FactGraph) -> ToolMask {
    let table = mood.table();
    let admitted: BTreeSet<&str> = table.allowed_tool_names().into_iter().collect();

    let mut prog = Gating::default();
    for t in available {
        prog.available.push((t.clone(),));
        if admitted.contains(t.as_str()) {
            prog.mood_admits.push((t.clone(),));
        }
    }

    for fact in graph.iter_live() {
        if *fact.relation() != Relation::IsA {
            continue;
        }
        match fact.object() {
            "read_only" => {
                for t in MUTATING_TOOLS {
                    prog.denied
                        .push(((*t).to_string(), "read_only".to_string()));
                }
            }
            "forbidden" => {
                if let Some(name) = fact.subject().strip_prefix("tool:") {
                    prog.denied
                        .push((name.to_string(), "forbidden_tool".to_string()));
                }
            }
            _ => {}
        }
    }

    prog.run();

    // BTreeSet rather than sort-after: the mask is rendered into the mood
    // segment on some paths, and duplicate or reordered entries would move
    // bytes in the cached prefix.
    let allowed: BTreeSet<String> = prog.allowed.iter().map(|(t,)| t.clone()).collect();
    let denied_by_mood: BTreeSet<String> = prog.mood_blocked.iter().map(|(t,)| t.clone()).collect();
    let denied_by_rule: BTreeSet<(String, String)> = prog
        .rule_blocked
        .iter()
        .map(|(t, r)| (t.clone(), r.clone()))
        .collect();

    ToolMask {
        allowed: allowed.into_iter().collect(),
        denied_by_mood: denied_by_mood.into_iter().collect(),
        denied_by_rule: denied_by_rule.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_factgraph::Scope;
    use qgi2_spec_types::{
        CommitToken, Confidence, ConflictPolicy, ProposedFact, Source,
    };

    fn tools() -> Vec<String> {
        ["read", "write", "edit", "bash", "browser", "gmail", "ls"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn graph_with(subject: &str, object: &str) -> FactGraph {
        let mut g = FactGraph::new();
        let f = ProposedFact {
            subject: subject.into(),
            relation: Relation::IsA,
            object: object.into(),
            confidence: Confidence::new(0.99),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::Rule("test".into()), 1);
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        g
    }

    #[test]
    fn builder_gets_filesystem_and_shell_not_mail() {
        let m = tool_mask(&tools(), Mood::Builder, &FactGraph::new());
        assert!(m.permits("read"));
        assert!(m.permits("bash"));
        assert!(!m.permits("gmail"));
        assert!(m.denied_by_mood.contains(&"gmail".to_string()));
    }

    #[test]
    fn researcher_gets_the_browser_not_the_shell() {
        let m = tool_mask(&tools(), Mood::Researcher, &FactGraph::new());
        assert!(m.permits("browser"));
        assert!(!m.permits("bash"));
        assert!(!m.permits("write"));
    }

    #[test]
    fn companion_gets_mail_not_the_filesystem() {
        let m = tool_mask(&tools(), Mood::Companion, &FactGraph::new());
        assert!(m.permits("gmail"));
        assert!(!m.permits("write"));
    }

    #[test]
    fn a_read_only_fact_removes_write_tools_without_changing_mood() {
        let g = graph_with("repo", "read_only");
        let m = tool_mask(&tools(), Mood::Builder, &g);
        assert!(m.permits("read"), "reads survive");
        assert!(m.permits("ls"));
        assert!(!m.permits("write"));
        assert!(!m.permits("edit"));
        assert!(!m.permits("bash"));
        assert_eq!(m.denial_reason("write").unwrap(), "denied by rule `read_only`");
    }

    #[test]
    fn a_forbidden_tool_fact_removes_exactly_that_tool() {
        let g = graph_with("tool:bash", "forbidden");
        let m = tool_mask(&tools(), Mood::Builder, &g);
        assert!(!m.permits("bash"));
        assert!(m.permits("write"), "other write tools are unaffected");
    }

    #[test]
    fn rule_denial_outranks_mood_in_the_reported_reason() {
        let g = graph_with("repo", "read_only");
        let m = tool_mask(&tools(), Mood::Builder, &g);
        assert!(m.denial_reason("bash").unwrap().contains("rule"));
        assert!(m.denial_reason("gmail").unwrap().contains("mood"));
        assert!(m.denial_reason("read").is_none());
    }

    #[test]
    fn a_tool_the_harness_does_not_offer_never_appears() {
        let m = tool_mask(&["read".to_string()], Mood::Builder, &FactGraph::new());
        assert_eq!(m.allowed, vec!["read".to_string()]);
        assert!(m.denied_by_mood.is_empty());
    }

    #[test]
    fn the_mask_is_sorted_and_deduplicated() {
        let dupes = vec!["read".to_string(), "read".to_string(), "bash".to_string()];
        let m = tool_mask(&dupes, Mood::Builder, &FactGraph::new());
        assert_eq!(m.allowed, vec!["bash".to_string(), "read".to_string()]);
    }
}
