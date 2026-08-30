//! The router: per-step `(model, speculation, sampling)` selection.
//!
//! Spec:
//!
//! > **Control layer** — Harness-owned router + assembler — Per-step
//! > model/spec/sampling selection; cache-aware prompt assembly with segment
//! > hashes.
//!
//! > Every step has an explicit `(model, speculation, sampling)` triple chosen
//! > by the router from mood and profile. **Nothing defaults.**
//!
//! "Nothing defaults" is enforced two ways. [`Router::plan`] returns
//! `Result`, not a `StepPlan` with fallbacks, so an unrouted step is a loud
//! error. And every plan is run through [`StepPlan::validate`] before it is
//! returned, so a table that pairs greedy sampling with DFlash2 — or forgets a
//! schema on a structured step — fails at the router rather than at the engine.

pub mod schemas;

use qgi2_spec_types::{
    Mood, ModelRole, Persona, Profile, Qgi2Error, Result, Sampling, Speculation, StepKind,
    StepPlan,
};

/// Chooses the triple for each step of a turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Router {
    pub persona: Persona,
}

impl Router {
    pub fn new(persona: Persona) -> Self {
        Self { persona }
    }

    pub fn mood(&self) -> Mood {
        self.persona.mood
    }

    pub fn profile(&self) -> Profile {
        self.persona.profile
    }

    /// The triple for one step.
    ///
    /// Model role comes from the spec's component table: the planner "plans and
    /// answers", the worker does "extract, render, route, tool-args". Sampling
    /// starts from the mood and is then overridden by the profile. Speculation
    /// is the profile's for worker steps, and MTP for the planner — the spec
    /// says "MTP on the planner", and unlike the worker the planner has no
    /// DFlash2 option to choose between.
    pub fn plan(&self, step: StepKind) -> Result<StepPlan> {
        if !step.is_model_step() {
            return Err(Qgi2Error::UnroutedStep {
                step,
                mood: self.mood(),
                profile: self.profile(),
            });
        }

        let role = match step {
            StepKind::Plan | StepKind::Answer => ModelRole::Planner,
            StepKind::Extract | StepKind::ToolArgs | StepKind::Route => ModelRole::Worker,
            // Guarded above, but matching exhaustively keeps a newly added rule
            // stage from silently acquiring a worker triple.
            StepKind::Verify | StepKind::Commit | StepKind::MoodCheck => {
                return Err(Qgi2Error::UnroutedStep {
                    step,
                    mood: self.mood(),
                    profile: self.profile(),
                });
            }
        };

        let mood_sampling = self.mood().table().planner_sampling;
        let mut sampling = self.profile().apply_sampling(mood_sampling);

        // The worker's steps are bookkeeping, not prose. Thinking on an
        // extraction spends planner-grade tokens on a step whose output is a
        // fixed-shape JSON object, and it pushes the planner:worker token ratio
        // the wrong way against the spec's 1:3 target.
        if role == ModelRole::Worker {
            sampling.thinking = false;
        }

        let speculation = self.speculation_for(role, step, &sampling);

        let plan = StepPlan {
            step,
            role,
            speculation,
            sampling,
            schema: schemas::for_step(step),
        };

        plan.validate().map_err(|detail| Qgi2Error::Schema {
            step,
            detail,
        })?;

        Ok(plan)
    }

    fn speculation_for(
        &self,
        role: ModelRole,
        step: StepKind,
        sampling: &Sampling,
    ) -> Speculation {
        match role {
            // Spec: "MTP on the planner". n=2 is the spec's planner example in
            // the per-turn loop (`plan [planner, MTP n=2, ...]`).
            ModelRole::Planner => Speculation::Mtp { n: 2 },
            ModelRole::Worker => {
                // Tool arguments largely copy identifiers out of the prompt —
                // file paths, symbol names — which is exactly the case the spec
                // reserves n-gram for. Only when the profile offers it.
                if step == StepKind::ToolArgs
                    && let Some(fallback) = self.profile().worker_speculation_fallback()
                {
                    return fallback;
                }
                let primary = self.profile().worker_speculation();
                // Belt and braces against a future table edit: a greedy
                // sampling must never be paired with a speculator that cannot
                // produce greedy output.
                if sampling.is_greedy() && !primary.supports_greedy() {
                    Speculation::Mtp { n: 3 }
                } else {
                    primary
                }
            }
        }
    }

    /// Every model step's plan, for logging the turn's routing up front.
    pub fn plan_all(&self) -> Result<Vec<StepPlan>> {
        StepKind::ALL
            .into_iter()
            .filter(|s| s.is_model_step())
            .map(|s| self.plan(s))
            .collect()
    }

    /// Which vLLM model name serves a role.
    ///
    /// The spec pins the planner to Qwen3.8-Flash-Next and the worker to
    /// Qwen3.8-27B, both NVFP4; the concrete served names are configuration.
    pub fn model_name(&self, role: ModelRole, config: &ModelNames) -> String {
        match role {
            ModelRole::Planner => config.planner.clone(),
            ModelRole::Worker => config.worker.clone(),
        }
    }
}

/// The names vLLM serves the two models under.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelNames {
    pub planner: String,
    pub worker: String,
}

impl Default for ModelNames {
    fn default() -> Self {
        Self {
            planner: "Qwen3.8-Flash-Next-NVFP4".to_string(),
            worker: "Qwen3.8-27B-NVFP4".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(mood: Mood, profile: Profile) -> Router {
        Router::new(Persona::new(mood, profile))
    }

    #[test]
    fn the_planner_plans_and_answers() {
        let r = router(Mood::Builder, Profile::Traceable);
        assert_eq!(r.plan(StepKind::Plan).unwrap().role, ModelRole::Planner);
        assert_eq!(r.plan(StepKind::Answer).unwrap().role, ModelRole::Planner);
    }

    #[test]
    fn the_worker_extracts_routes_and_fills_tool_args() {
        let r = router(Mood::Builder, Profile::Traceable);
        for s in [StepKind::Extract, StepKind::ToolArgs, StepKind::Route] {
            assert_eq!(r.plan(s).unwrap().role, ModelRole::Worker, "{s}");
        }
    }

    #[test]
    fn rule_stages_are_unroutable() {
        let r = router(Mood::Builder, Profile::Traceable);
        for s in [StepKind::Verify, StepKind::Commit, StepKind::MoodCheck] {
            assert!(
                matches!(r.plan(s), Err(Qgi2Error::UnroutedStep { .. })),
                "{s} should not route"
            );
        }
    }

    #[test]
    fn the_planner_always_speculates_with_mtp() {
        for mood in Mood::ALL {
            for profile in Profile::ALL {
                let p = router(mood, profile).plan(StepKind::Plan).unwrap();
                assert!(
                    matches!(p.speculation, Speculation::Mtp { .. }),
                    "{mood}/{profile} gave the planner {:?}",
                    p.speculation
                );
            }
        }
    }

    #[test]
    fn mood_sets_the_planner_temperature() {
        assert_eq!(
            router(Mood::Builder, Profile::Traceable)
                .plan(StepKind::Plan)
                .unwrap()
                .sampling
                .temperature,
            0.3
        );
        assert_eq!(
            router(Mood::Researcher, Profile::Traceable)
                .plan(StepKind::Plan)
                .unwrap()
                .sampling
                .temperature,
            0.7
        );
    }

    #[test]
    fn the_deterministic_profile_forces_greedy_and_a_greedy_capable_speculator() {
        for mood in Mood::ALL {
            for step in [StepKind::Plan, StepKind::Extract, StepKind::Answer] {
                let p = router(mood, Profile::Deterministic).plan(step).unwrap();
                assert!(p.sampling.is_greedy(), "{mood}/{step}");
                assert!(p.sampling.seed.is_some());
                assert!(p.sampling.batch_invariant);
                assert!(
                    p.speculation.supports_greedy(),
                    "{mood}/{step} paired greedy with {:?}",
                    p.speculation
                );
            }
        }
    }

    #[test]
    fn quick_uses_ngram_for_tool_arguments() {
        // Tool args copy identifiers out of the prompt, which is the case the
        // spec reserves n-gram for.
        let p = router(Mood::Builder, Profile::Quick)
            .plan(StepKind::ToolArgs)
            .unwrap();
        assert!(matches!(p.speculation, Speculation::NGram { .. }), "{:?}", p.speculation);
    }

    #[test]
    fn traceable_keeps_one_speculator_for_the_worker() {
        // Traceable measures acceptance; swapping methods per step would make
        // the number meaningless.
        let r = router(Mood::Builder, Profile::Traceable);
        for s in [StepKind::Extract, StepKind::ToolArgs, StepKind::Route] {
            assert_eq!(r.plan(s).unwrap().speculation, Speculation::DFlash2 { n: 7 }, "{s}");
        }
    }

    #[test]
    fn worker_steps_never_think() {
        // Thinking on a fixed-shape JSON step spends planner-grade tokens and
        // pushes the planner:worker ratio the wrong way.
        for mood in Mood::ALL {
            for profile in Profile::ALL {
                let p = router(mood, profile).plan(StepKind::Extract).unwrap();
                assert!(!p.sampling.thinking, "{mood}/{profile}");
            }
        }
    }

    #[test]
    fn quick_turns_off_planner_thinking() {
        assert!(
            !router(Mood::Builder, Profile::Quick)
                .plan(StepKind::Plan)
                .unwrap()
                .sampling
                .thinking
        );
        assert!(
            router(Mood::Builder, Profile::Traceable)
                .plan(StepKind::Plan)
                .unwrap()
                .sampling
                .thinking
        );
    }

    #[test]
    fn every_structured_step_carries_a_schema_and_answer_does_not() {
        let r = router(Mood::Builder, Profile::Traceable);
        for s in [StepKind::Plan, StepKind::Extract, StepKind::ToolArgs, StepKind::Route] {
            assert!(r.plan(s).unwrap().schema.is_some(), "{s}");
        }
        assert!(r.plan(StepKind::Answer).unwrap().schema.is_none());
    }

    #[test]
    fn every_mood_and_profile_combination_routes_every_model_step() {
        // "Nothing defaults" also means nothing is missing.
        for mood in Mood::ALL {
            for profile in Profile::ALL {
                let plans = router(mood, profile).plan_all().unwrap();
                assert_eq!(plans.len(), 5, "{mood}/{profile}");
                for p in plans {
                    p.validate().unwrap_or_else(|e| panic!("{mood}/{profile}: {e}"));
                }
            }
        }
    }
}
