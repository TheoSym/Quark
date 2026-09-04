//! Skill selection.
//!
//! jcode already activates skills on an embedding hit. QGI-2 adds a rule layer
//! on top: a skill can also be activated because the *graph* says it is
//! relevant, which catches the case where the query text does not resemble the
//! skill description but the retrieved subgraph does.
//!
//! Selected skills are rendered into segment 4, which is volatile — so the
//! selection changing between turns costs only the tail, not the cached prefix.
//! It is still worth keeping the set small and stable: a skill that flickers on
//! and off every turn recomputes segment 4 for no benefit.
//!
//! The rules, each a named function below. The last one is the only recursive
//! rule in the crate — requirements close transitively — and is a worklist.
//!
//! ```text
//! mood_ok(s) <- skill(s), !mood_scoped(s)
//! mood_ok(s) <- applies_to(s, m), current_mood(m)
//! matched(s) <- covers(s, p), reached(n), n starts_with p
//! active(s)  <- matched(s), mood_ok(s)
//! active(s)  <- forced(s), mood_ok(s)
//! active(r)  <- active(s), requires(s, r), mood_ok(r)
//! ```

use qgi2_factgraph::FactGraph;
use qgi2_spec_types::{Mood, Relation};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A skill the harness could activate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillCandidate {
    pub name: String,
    /// Node-name prefixes this skill covers, e.g. `file:` or `task:deploy`.
    pub subjects: Vec<String>,
    /// Moods this skill applies to. Empty means every mood.
    pub moods: Vec<Mood>,
    /// Skills that must also activate when this one does.
    pub requires: Vec<String>,
}

impl SkillCandidate {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subjects: Vec::new(),
            moods: Vec::new(),
            requires: Vec::new(),
        }
    }

    pub fn covering(mut self, prefixes: &[&str]) -> Self {
        self.subjects = prefixes.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn for_moods(mut self, moods: &[Mood]) -> Self {
        self.moods = moods.to_vec();
        self
    }

    pub fn requiring(mut self, names: &[&str]) -> Self {
        self.requires = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// `mood_ok`: the skill's mood restriction, if any, is satisfied.
    fn mood_ok(&self, mood: Mood) -> bool {
        self.moods.is_empty() || self.moods.contains(&mood)
    }

    /// `matched`: some reached node starts with one of the covered prefixes.
    fn matched(&self, reached: &[String]) -> bool {
        self.subjects
            .iter()
            .any(|p| reached.iter().any(|n| n.starts_with(p.as_str())))
    }
}

/// Choose which skills to render into segment 4.
///
/// `reached` is the set of node names the turn's retrieval touched;
/// `forced` are skills the user activated by slash command or tool call.
///
/// A forced skill still respects its own mood restriction — activating a
/// Builder-only skill in Companion mood would put instructions in the prompt
/// that contradict the mood segment. Requirements activate transitively and
/// are themselves mood-gated; a requirement naming a skill that is not a
/// candidate activates nothing.
pub fn select_skills(
    candidates: &[SkillCandidate],
    reached: &[String],
    mood: Mood,
    forced: &[String],
    _graph: &FactGraph,
) -> Vec<String> {
    let by_name: BTreeMap<&str, &SkillCandidate> =
        candidates.iter().map(|c| (c.name.as_str(), c)).collect();
    let forced: BTreeSet<&str> = forced.iter().map(String::as_str).collect();

    // Seed: active(s) <- matched(s), mood_ok(s)  |  forced(s), mood_ok(s)
    let mut active: BTreeSet<&str> = BTreeSet::new();
    let mut worklist: Vec<&SkillCandidate> = Vec::new();
    for c in candidates {
        if c.mood_ok(mood) && (c.matched(reached) || forced.contains(c.name.as_str())) {
            active.insert(c.name.as_str());
            worklist.push(c);
        }
    }

    // Closure: active(r) <- active(s), requires(s, r), mood_ok(r)
    while let Some(s) = worklist.pop() {
        for r in &s.requires {
            if let Some(req) = by_name.get(r.as_str())
                && req.mood_ok(mood)
                && active.insert(req.name.as_str())
            {
                worklist.push(req);
            }
        }
    }

    // Sorted: segment 4 is regenerated each turn, and an unstable order would
    // make it differ even when the selected set did not.
    active.into_iter().map(str::to_string).collect()
}

/// Node names a set of facts touched, for use as `reached`.
pub fn reached_nodes(graph: &FactGraph, relation_filter: Option<&Relation>) -> Vec<String> {
    let mut out = BTreeSet::new();
    for f in graph.iter_live() {
        if let Some(r) = relation_filter
            && f.relation() != r
        {
            continue;
        }
        out.insert(f.subject().to_string());
        out.insert(f.object().to_string());
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<SkillCandidate> {
        vec![
            SkillCandidate::new("rust-review").covering(&["file:", "task:"]),
            SkillCandidate::new("citation-check")
                .covering(&["claim:"])
                .for_moods(&[Mood::Researcher]),
            SkillCandidate::new("deploy").covering(&["task:deploy"]).requiring(&["rust-review"]),
            SkillCandidate::new("never-matches").covering(&["zzz:"]),
        ]
    }

    fn g() -> FactGraph {
        FactGraph::new()
    }

    #[test]
    fn a_reached_node_activates_the_covering_skill() {
        let out = select_skills(&candidates(), &["file:auth.rs".into()], Mood::Builder, &[], &g());
        assert!(out.contains(&"rust-review".to_string()));
        assert!(!out.contains(&"never-matches".to_string()));
    }

    #[test]
    fn mood_scoped_skills_stay_out_of_the_wrong_mood() {
        let reached = vec!["claim:x".to_string()];
        let builder = select_skills(&candidates(), &reached, Mood::Builder, &[], &g());
        assert!(!builder.contains(&"citation-check".to_string()));
        let researcher = select_skills(&candidates(), &reached, Mood::Researcher, &[], &g());
        assert!(researcher.contains(&"citation-check".to_string()));
    }

    #[test]
    fn requirements_activate_transitively() {
        let out = select_skills(&candidates(), &["task:deploy".into()], Mood::Builder, &[], &g());
        assert!(out.contains(&"deploy".to_string()));
        assert!(out.contains(&"rust-review".to_string()), "requirement pulled in");
    }

    #[test]
    fn requirement_chains_close_and_cycles_terminate() {
        // a -> b -> c, and c -> a: the closure must reach c and must not loop.
        let cands = vec![
            SkillCandidate::new("a").covering(&["x:"]).requiring(&["b"]),
            SkillCandidate::new("b").requiring(&["c"]),
            SkillCandidate::new("c").requiring(&["a"]),
        ];
        let out = select_skills(&cands, &["x:1".into()], Mood::Builder, &[], &g());
        assert_eq!(out, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn a_mood_gated_requirement_breaks_the_chain() {
        // a -> b (Researcher only) -> c. In Builder, b is not mood_ok, so
        // neither b nor anything only reachable through it activates.
        let cands = vec![
            SkillCandidate::new("a").covering(&["x:"]).requiring(&["b"]),
            SkillCandidate::new("b").for_moods(&[Mood::Researcher]).requiring(&["c"]),
            SkillCandidate::new("c"),
        ];
        let out = select_skills(&cands, &["x:1".into()], Mood::Builder, &[], &g());
        assert_eq!(out, vec!["a".to_string()]);
    }

    #[test]
    fn a_requirement_that_is_not_a_candidate_activates_nothing() {
        let cands = vec![SkillCandidate::new("a").covering(&["x:"]).requiring(&["ghost"])];
        let out = select_skills(&cands, &["x:1".into()], Mood::Builder, &[], &g());
        assert_eq!(out, vec!["a".to_string()]);
    }

    #[test]
    fn forcing_activates_a_skill_nothing_reached() {
        let out = select_skills(&candidates(), &[], Mood::Builder, &["never-matches".into()], &g());
        assert_eq!(out, vec!["never-matches".to_string()]);
    }

    #[test]
    fn forcing_still_respects_a_mood_restriction() {
        // Otherwise a forced Researcher skill injects instructions that
        // contradict the Builder mood segment sitting above it in the prompt.
        let out = select_skills(&candidates(), &[], Mood::Builder, &["citation-check".into()], &g());
        assert!(out.is_empty(), "got {out:?}");
    }

    #[test]
    fn forcing_an_unknown_skill_activates_nothing() {
        let out = select_skills(&candidates(), &[], Mood::Builder, &["ghost".into()], &g());
        assert!(out.is_empty(), "got {out:?}");
    }

    #[test]
    fn selection_is_sorted_and_deduplicated() {
        let out = select_skills(
            &candidates(),
            &["file:a".into(), "task:b".into(), "task:deploy".into()],
            Mood::Builder,
            &["rust-review".into()],
            &g(),
        );
        let mut sorted = out.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(out, sorted);
    }

    #[test]
    fn nothing_reached_activates_nothing() {
        assert!(select_skills(&candidates(), &[], Mood::Builder, &[], &g()).is_empty());
    }

    #[test]
    fn prefix_matching_is_a_prefix_not_a_substring() {
        // "task:deploy" covers "task:deployment" but not "old-task:deploy".
        let out = select_skills(&candidates(), &["old-task:deploy".into()], Mood::Builder, &[], &g());
        assert!(!out.contains(&"deploy".to_string()));
    }
}
