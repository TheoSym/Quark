//! Mood switching.
//!
//! The per-turn loop ends with `mood check [rules]`. A mood switch is
//! expensive: segment 2 is part of the byte-stable cached prefix, so changing
//! mood invalidates everything after it and forces a full prefill on the next
//! turn.
//!
//! That cost is why this module is conservative. A switch needs *sustained*
//! evidence — a share of recent extractions belonging to another mood's
//! traversal set, over a minimum number of observations — rather than a single
//! off-mood fact. [`MoodDecision`] always reports the cache consequence so the
//! caller can surface it rather than discovering it in the next turn's hit
//! rate.

use qgi2_spec_types::{Mood, Relation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How eager the mood check is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoodSwitchConfig {
    /// Fraction of recent relations that must belong to the target mood.
    pub threshold: f32,
    /// Minimum observations before a switch is considered at all.
    pub min_observations: usize,
    /// The target must beat the incumbent by at least this margin, so a mood
    /// that is barely ahead does not cause repeated flip-flopping.
    pub margin: f32,
}

impl Default for MoodSwitchConfig {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_observations: 5,
            margin: 0.2,
        }
    }
}

/// The outcome of a mood check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoodDecision {
    /// No change.
    Stay {
        mood: Mood,
        reason: String,
    },
    /// Switch, at the cost of the cached prefix from segment 2 onward.
    Switch {
        from: Mood,
        to: Mood,
        share: f32,
        /// Always true — segment 2 is in the stable prefix. Carried explicitly
        /// so the caller reports the cache cost instead of silently paying it.
        invalidates_prefix: bool,
        reason: String,
    },
}

impl MoodDecision {
    pub fn mood(&self) -> Mood {
        match self {
            Self::Stay { mood, .. } => *mood,
            Self::Switch { to, .. } => *to,
        }
    }

    pub fn is_switch(&self) -> bool {
        matches!(self, Self::Switch { .. })
    }
}

ascent::ascent! {
    /// Which moods the recently observed relations belong to.
    pub struct MoodEvidence;

    /// (relation, mood) — the mood's traversal set contains this relation
    relation belongs(String, String);
    /// A relation observed this session, once per observation
    relation observed(usize, String);

    /// (observation index, mood) — this observation is evidence for that mood
    relation evidence(usize, String);
    evidence(i, m) <-- observed(i, r), belongs(r, m);
}

/// Decide whether to switch mood.
///
/// `recent` is the relations extracted over the recent window, most recent
/// last. A relation shared by several moods (none are, in the default tables,
/// but a configured mood could overlap) counts as evidence for each.
pub fn mood_check(
    current: Mood,
    recent: &[Relation],
    config: MoodSwitchConfig,
) -> MoodDecision {
    if recent.len() < config.min_observations {
        return MoodDecision::Stay {
            mood: current,
            reason: format!(
                "only {} observations; {} needed before a switch is considered",
                recent.len(),
                config.min_observations
            ),
        };
    }

    let mut prog = MoodEvidence::default();
    for m in Mood::ALL {
        for r in &m.table().traversal.relations {
            prog.belongs
                .push((r.as_str().to_string(), m.as_str().to_string()));
        }
    }
    for (i, r) in recent.iter().enumerate() {
        prog.observed.push((i, r.as_str().to_string()));
    }
    prog.run();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, m) in prog.evidence.iter() {
        *counts.entry(m.clone()).or_insert(0) += 1;
    }

    let total = recent.len() as f32;
    let share = |m: Mood| counts.get(m.as_str()).copied().unwrap_or(0) as f32 / total;
    let current_share = share(current);

    // Deterministic ordering: sort by share descending, then by mood name, so
    // a tie does not depend on map iteration.
    let mut ranked: Vec<(Mood, f32)> = Mood::ALL.into_iter().map(|m| (m, share(m))).collect();
    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.as_str().cmp(b.0.as_str()))
    });

    let (best, best_share) = ranked[0];
    if best == current {
        return MoodDecision::Stay {
            mood: current,
            reason: format!("{current} still leads at {:.0}% of recent relations", best_share * 100.0),
        };
    }
    if best_share < config.threshold {
        return MoodDecision::Stay {
            mood: current,
            reason: format!(
                "{best} leads at {:.0}% but the threshold is {:.0}%",
                best_share * 100.0,
                config.threshold * 100.0
            ),
        };
    }
    if best_share - current_share < config.margin {
        return MoodDecision::Stay {
            mood: current,
            reason: format!(
                "{best} at {:.0}% does not beat {current} at {:.0}% by the {:.0}% margin",
                best_share * 100.0,
                current_share * 100.0,
                config.margin * 100.0
            ),
        };
    }

    MoodDecision::Switch {
        from: current,
        to: best,
        share: best_share,
        invalidates_prefix: true,
        reason: format!(
            "{:.0}% of the last {} relations belong to {best}",
            best_share * 100.0,
            recent.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(r: Relation, n: usize) -> Vec<Relation> {
        std::iter::repeat_n(r, n).collect()
    }

    #[test]
    fn too_few_observations_never_switches() {
        let d = mood_check(Mood::Builder, &rep(Relation::Supports, 2), MoodSwitchConfig::default());
        assert!(!d.is_switch());
        assert_eq!(d.mood(), Mood::Builder);
    }

    #[test]
    fn sustained_off_mood_evidence_switches() {
        let d = mood_check(
            Mood::Builder,
            &rep(Relation::Supports, 10),
            MoodSwitchConfig::default(),
        );
        assert!(d.is_switch(), "{d:?}");
        assert_eq!(d.mood(), Mood::Researcher);
    }

    #[test]
    fn a_switch_always_reports_the_cache_cost() {
        // Segment 2 sits in the stable prefix; a switch that did not say so
        // would show up only as a mysterious cache-hit drop next turn.
        let d = mood_check(
            Mood::Builder,
            &rep(Relation::Prefers, 10),
            MoodSwitchConfig::default(),
        );
        match d {
            MoodDecision::Switch { invalidates_prefix, to, .. } => {
                assert!(invalidates_prefix);
                assert_eq!(to, Mood::Companion);
            }
            other => panic!("expected a switch, got {other:?}"),
        }
    }

    #[test]
    fn a_sub_threshold_lead_does_not_switch() {
        // 6 researcher / 5 builder: researcher leads at 55%, under the 60%
        // threshold.
        let mut recent = rep(Relation::Supports, 6);
        recent.extend(rep(Relation::DependsOn, 5));
        let d = mood_check(Mood::Builder, &recent, MoodSwitchConfig::default());
        assert!(!d.is_switch(), "{d:?}");
        match d {
            MoodDecision::Stay { reason, .. } => assert!(reason.contains("threshold")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_margin_guards_against_flip_flopping() {
        // With the default threshold every observation maps to exactly one
        // mood, so clearing 60% already implies a 20-point lead and the margin
        // never binds. It exists for configurations that lower the threshold,
        // where two moods can sit close together and a switch each turn would
        // rebuild the cached prefix repeatedly.
        let config = MoodSwitchConfig {
            threshold: 0.4,
            min_observations: 5,
            margin: 0.3,
        };
        let mut recent = rep(Relation::Supports, 5);
        recent.extend(rep(Relation::DependsOn, 4));
        let d = mood_check(Mood::Builder, &recent, config);
        assert!(!d.is_switch(), "{d:?}");
        match d {
            MoodDecision::Stay { reason, .. } => assert!(reason.contains("margin")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn staying_in_the_leading_mood_is_reported_as_such() {
        let d = mood_check(
            Mood::Builder,
            &rep(Relation::DependsOn, 10),
            MoodSwitchConfig::default(),
        );
        assert!(!d.is_switch());
        match d {
            MoodDecision::Stay { reason, .. } => assert!(reason.contains("still leads")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn structural_relations_are_evidence_for_no_mood() {
        // is_a / part_of belong to no traversal set, so a session full of them
        // must not drag the mood anywhere.
        let d = mood_check(Mood::Builder, &rep(Relation::IsA, 10), MoodSwitchConfig::default());
        assert!(!d.is_switch(), "{d:?}");
    }

    #[test]
    fn the_decision_is_deterministic() {
        let recent = rep(Relation::Supports, 10);
        let a = mood_check(Mood::Builder, &recent, MoodSwitchConfig::default());
        let b = mood_check(Mood::Builder, &recent, MoodSwitchConfig::default());
        assert_eq!(a, b);
    }
}
