//! Compiled Datalog rules.
//!
//! Spec:
//!
//! > Rules — Compiled Datalog in Rust (`crepe`/`ascent`) — Retrieval traversal,
//! > tool gating, consistency, skill selection, mood switching.
//!
//! and the invariant this crate exists to enforce:
//!
//! > The model proposes facts; rules validate and commit. The graph is never
//! > written by the model directly.
//!
//! `qgi2-spec-types` makes that structural: a [`ProposedFact`] can only become
//! a committed [`Fact`] by way of a `CommitToken`, and this is the only crate
//! that mints one. If you find yourself wanting a token elsewhere, the thing
//! you actually want is a rule here.
//!
//! Confidence is carried through the Datalog programs as a scaled `u32`
//! (`confidence * 1000`) because ascent relations need `Eq + Hash`, which `f32`
//! is not. [`scale`] and [`unscale`] are the only places that conversion
//! happens.

pub mod gating;
pub mod mood_switch;
pub mod skills;
pub mod verify;

pub use gating::{ToolMask, tool_mask};
pub use mood_switch::{MoodDecision, MoodSwitchConfig, mood_check};
pub use skills::{SkillCandidate, reached_nodes, select_skills};
pub use verify::{Rejection, RejectionReason, VerifyConfig, VerifyOutcome, verify};

/// Scale a confidence into the integer domain the Datalog programs use.
pub(crate) fn scale(c: f32) -> u32 {
    (c.clamp(0.0, 1.0) * 1000.0).round() as u32
}

/// Inverse of [`scale`].
pub(crate) fn unscale(v: u32) -> f32 {
    v as f32 / 1000.0
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    #[test]
    fn scaling_round_trips_within_one_thousandth() {
        for c in [0.0f32, 0.25, 0.5, 0.755, 0.999, 1.0] {
            assert!((unscale(scale(c)) - c).abs() <= 0.001, "{c}");
        }
    }

    #[test]
    fn scaling_saturates_out_of_range_input() {
        assert_eq!(scale(2.0), 1000);
        assert_eq!(scale(-1.0), 0);
    }
}
