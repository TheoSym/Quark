//! Moods: config over one core.
//!
//! Transcribed from the spec's mood table:
//!
//! | | Builder | Researcher | Companion |
//! |---|---|---|---|
//! | Traversal | Task→depends_on→File | Claim→supports/contradicts→Source | Person→prefers→Topic |
//! | Conflict | latest wins | keep both | highest confidence |
//! | Tools | fs, shell, git | web, fetch, docs | calendar, mail, notes |
//! | Planner sampling | T 0.3 | T 0.7 | T 0.7 |
//! | Worker spec | DFlash2 | DFlash2 | DFlash2 |
//!
//! The table is data, not branches: [`Mood::table`] returns the row, and the
//! router and rules read fields off it. Adding a mood means adding a row.

use crate::fact::Relation;
use std::borrow::Cow;
use crate::step::{Sampling, Speculation};
use serde::{Deserialize, Serialize};
use std::fmt;

/// What the agent is currently doing. Orthogonal to [`crate::Profile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mood {
    Builder,
    Researcher,
    Companion,
}

impl Mood {
    pub const ALL: [Mood; 3] = [Mood::Builder, Mood::Researcher, Mood::Companion];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Builder => "builder",
            Self::Researcher => "researcher",
            Self::Companion => "companion",
        }
    }

    /// The spec's table row for this mood.
    pub fn table(self) -> MoodTable {
        match self {
            Self::Builder => MoodTable {
                mood: self,
                traversal: TraversalSpec {
                    root_type: Cow::Borrowed("Task"),
                    relations: vec![Relation::DependsOn, Relation::Implements, Relation::Modifies],
                    leaf_type: Cow::Borrowed("File"),
                },
                conflict: ConflictPolicy::LatestWins,
                tools: vec![ToolClass::Fs, ToolClass::Shell, ToolClass::Git],
                planner_sampling: Sampling::at_temperature(0.3),
                worker_speculation: Speculation::DFlash2 { n: 7 },
            },
            Self::Researcher => MoodTable {
                mood: self,
                traversal: TraversalSpec {
                    root_type: Cow::Borrowed("Claim"),
                    relations: vec![Relation::Supports, Relation::Contradicts, Relation::CitedBy],
                    leaf_type: Cow::Borrowed("Source"),
                },
                conflict: ConflictPolicy::KeepBoth,
                tools: vec![ToolClass::Web, ToolClass::Fetch, ToolClass::Docs],
                planner_sampling: Sampling::at_temperature(0.7),
                worker_speculation: Speculation::DFlash2 { n: 7 },
            },
            Self::Companion => MoodTable {
                mood: self,
                traversal: TraversalSpec {
                    root_type: Cow::Borrowed("Person"),
                    relations: vec![Relation::Prefers, Relation::Dislikes, Relation::KnowsAbout],
                    leaf_type: Cow::Borrowed("Topic"),
                },
                conflict: ConflictPolicy::HighestConfidence,
                tools: vec![ToolClass::Calendar, ToolClass::Mail, ToolClass::Notes],
                planner_sampling: Sampling::at_temperature(0.7),
                worker_speculation: Speculation::DFlash2 { n: 7 },
            },
        }
    }
}

impl fmt::Display for Mood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Mood {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "builder" => Ok(Self::Builder),
            "researcher" => Ok(Self::Researcher),
            "companion" => Ok(Self::Companion),
            other => Err(format!(
                "unknown mood {other:?}; expected builder, researcher, or companion"
            )),
        }
    }
}

/// How the retrieval traversal walks the graph for a mood.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraversalSpec {
    /// Node type traversal starts from, e.g. `Task`.
    pub root_type: Cow<'static, str>,
    /// Relations followed, in priority order.
    pub relations: Vec<Relation>,
    /// Node type traversal is trying to reach, e.g. `File`.
    pub leaf_type: Cow<'static, str>,
}

impl TraversalSpec {
    pub fn follows(&self, r: &Relation) -> bool {
        self.relations.contains(r)
    }
}

/// What to do when two facts with the same subject and relation disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictPolicy {
    /// Builder: the newer fact supersedes the older one. Code moves on.
    LatestWins,
    /// Researcher: contradictory claims are both evidence. Keep both and let
    /// the answer step present the disagreement.
    KeepBoth,
    /// Companion: keep whichever the agent is more sure of.
    HighestConfidence,
}

/// Coarse tool families a mood admits. The rules layer turns these into the
/// concrete allow-mask over the harness's actual tool names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolClass {
    Fs,
    Shell,
    Git,
    Web,
    Fetch,
    Docs,
    Calendar,
    Mail,
    Notes,
}

impl ToolClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fs => "fs",
            Self::Shell => "shell",
            Self::Git => "git",
            Self::Web => "web",
            Self::Fetch => "fetch",
            Self::Docs => "docs",
            Self::Calendar => "calendar",
            Self::Mail => "mail",
            Self::Notes => "notes",
        }
    }

    /// jcode tool names that belong to this class.
    ///
    /// This is the one place QGI-2 names jcode's tools. It is a lookup table
    /// rather than a change to jcode: the gating happens on QGI-2's side of the
    /// seam, by masking the tool list before it reaches the model.
    pub fn jcode_tools(self) -> &'static [&'static str] {
        match self {
            Self::Fs => &["read", "write", "edit", "multiedit", "patch", "apply_patch", "ls", "agentgrep"],
            Self::Shell => &["bash", "bg"],
            Self::Git => &["bash"],
            Self::Web => &["browser", "websearch"],
            Self::Fetch => &["webfetch", "open"],
            Self::Docs => &["jcode_docs", "skill"],
            Self::Calendar => &["calendar"],
            Self::Mail => &["gmail"],
            Self::Notes => &["memory", "side_panel"],
        }
    }
}

/// One row of the mood table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoodTable {
    pub mood: Mood,
    pub traversal: TraversalSpec,
    pub conflict: ConflictPolicy,
    pub tools: Vec<ToolClass>,
    pub planner_sampling: Sampling,
    pub worker_speculation: Speculation,
}

impl MoodTable {
    /// Concrete jcode tool names this mood admits, deduplicated and sorted.
    ///
    /// Sorted because this list is rendered into the mood segment, which the
    /// spec requires to be byte-stable within a session.
    pub fn allowed_tool_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self
            .tools
            .iter()
            .flat_map(|c| c.jcode_tools().iter().copied())
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// The mood segment's rendered text.
    ///
    /// Deterministic: same mood, same bytes, every session. That is what keeps
    /// segment 2 in the cacheable prefix.
    pub fn render(&self) -> String {
        let relations = self
            .traversal
            .relations
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join("/");
        format!(
            "# Mood: {mood}\n\
             Traversal: {root}-[{relations}]->{leaf}\n\
             Conflict policy: {conflict}\n\
             Tools: {tools}",
            mood = self.mood,
            root = self.traversal.root_type,
            leaf = self.traversal.leaf_type,
            conflict = match self.conflict {
                ConflictPolicy::LatestWins => "latest wins",
                ConflictPolicy::KeepBoth => "keep both",
                ConflictPolicy::HighestConfidence => "highest confidence",
            },
            tools = self
                .tools
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mood_table_matches_the_spec() {
        let b = Mood::Builder.table();
        assert_eq!(b.conflict, ConflictPolicy::LatestWins);
        assert_eq!(b.planner_sampling.temperature, 0.3);
        assert!(b.traversal.follows(&Relation::DependsOn));

        let r = Mood::Researcher.table();
        assert_eq!(r.conflict, ConflictPolicy::KeepBoth);
        assert_eq!(r.planner_sampling.temperature, 0.7);
        assert!(r.traversal.follows(&Relation::Contradicts));

        let c = Mood::Companion.table();
        assert_eq!(c.conflict, ConflictPolicy::HighestConfidence);
        assert_eq!(c.planner_sampling.temperature, 0.7);
        assert!(c.traversal.follows(&Relation::Prefers));
    }

    #[test]
    fn every_mood_uses_dflash2_for_the_worker() {
        // The spec's worker-spec row is DFlash2 across all three moods; the
        // profile, not the mood, is what can change it.
        for m in Mood::ALL {
            assert!(matches!(m.table().worker_speculation, Speculation::DFlash2 { .. }));
        }
    }

    #[test]
    fn mood_render_is_byte_stable() {
        for m in Mood::ALL {
            assert_eq!(m.table().render(), m.table().render());
        }
    }

    #[test]
    fn allowed_tool_names_are_sorted_and_deduped() {
        // Builder maps both `shell` and `git` onto `bash`; the mood segment
        // must not render it twice or the bytes depend on class order.
        let names = Mood::Builder.table().allowed_tool_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(names.iter().filter(|n| **n == "bash").count(), 1);
    }

    #[test]
    fn moods_round_trip_through_strings() {
        for m in Mood::ALL {
            assert_eq!(m.as_str().parse::<Mood>().unwrap(), m);
        }
    }
}
