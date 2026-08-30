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

use qgi2_factgraph::FactGraph;
use qgi2_spec_types::{Mood, Relation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
}

ascent::ascent! {
    /// Skill activation, including transitive requirements.
    pub struct SkillSelection;

    /// A skill that exists.
    relation skill(String);
    /// (skill, subject prefix) it covers
    relation covers(String, String);
    /// A skill restricted to a mood; absent means unrestricted
    relation mood_scoped(String);
    /// (skill, mood) pairs for scoped skills
    relation applies_to(String, String);
    /// The mood in effect this turn
    relation current_mood(String);
    /// A node the retrieval reached this turn
    relation reached(String);
    /// (skill, required skill)
    relation requires(String, String);
    /// A skill the user activated explicitly
    relation forced(String);

    /// A skill whose mood restriction, if any, is satisfied.
    relation mood_ok(String);
    mood_ok(s) <-- skill(s), !mood_scoped(s);
    mood_ok(s) <-- applies_to(s, m), current_mood(m);

    /// A skill matched by a reached node.
    relation matched(String);
    matched(s) <-- covers(s, p), reached(n), if n.starts_with(p.as_str());

    /// Activated: matched or forced, and mood-compatible. A forced skill still
    /// respects its own mood restriction — activating a Builder-only skill in
    /// Companion mood would put instructions in the prompt that contradict the
    /// mood segment.
    relation active(String);
    active(s) <-- matched(s), mood_ok(s);
    active(s) <-- forced(s), mood_ok(s);

    // Requirements activate transitively, and are themselves mood-gated.
    active(r) <-- active(s), requires(s, r), mood_ok(r);
}

/// Choose which skills to render into segment 4.
///
/// `reached` is the set of node names the turn's retrieval touched;
/// `forced` are skills the user activated by slash command or tool call.
pub fn select_skills(
    candidates: &[SkillCandidate],
    reached: &[String],
    mood: Mood,
    forced: &[String],
    _graph: &FactGraph,
) -> Vec<String> {
    let mut prog = SkillSelection::default();

    for c in candidates {
        prog.skill.push((c.name.clone(),));
        for p in &c.subjects {
            prog.covers.push((c.name.clone(), p.clone()));
        }
        if !c.moods.is_empty() {
            prog.mood_scoped.push((c.name.clone(),));
            for m in &c.moods {
                prog.applies_to
                    .push((c.name.clone(), m.as_str().to_string()));
            }
        }
        for r in &c.requires {
            prog.requires.push((c.name.clone(), r.clone()));
        }
    }

    prog.current_mood.push((mood.as_str().to_string(),));
    for n in reached {
        prog.reached.push((n.clone(),));
    }
    for f in forced {
        prog.forced.push((f.clone(),));
    }

    prog.run();

    // Sorted: segment 4 is regenerated each turn, and an unstable order would
    // make it differ even when the selected set did not.
    let active: BTreeSet<String> = prog.active.iter().map(|(s,)| s.clone()).collect();
    active.into_iter().collect()
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
