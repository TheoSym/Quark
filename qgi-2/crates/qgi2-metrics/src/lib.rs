//! Per-turn metrics.
//!
//! Spec success metrics:
//!
//! - Prefix-cache hit rate per model >= 85% (from `cached_tokens`).
//! - Speculation acceptance: planner MTP >= 1.8 tokens/step; worker DFlash2 >= 2.0.
//! - Tokens per turn trending down as memory replaces raw context.
//! - Extraction rejection rate at verify <= 10%.
//! - Planner:worker token ratio <= 1:3.
//!
//! and, crucially:
//!
//! > Cache hit rate is measured every turn from vLLM's `cached_tokens` and
//! > surfaced in the UI. **A drop below threshold is a bug, not a metric.**
//!
//! So [`TurnMetrics::breaches`] returns findings phrased as defects, and the
//! turn loop surfaces them rather than filing them away. Nothing here silently
//! adapts a threshold to what the system happens to be achieving.
//!
//! Session end writes these back into the graph as [`Source::Harness`] facts,
//! which is the spec's self-tuning loop: "log speculation acceptance and
//! cache-hit stats as facts".

use qgi2_spec_types::{
    Confidence, Fact, FactKey, ModelRole, Relation, Source, Thresholds, TurnIndex,
};
use serde::{Deserialize, Serialize};

/// What one turn cost and how well it cached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnMetrics {
    pub turn: TurnIndex,
    pub planner_prompt_tokens: u64,
    pub planner_completion_tokens: u64,
    pub planner_cached_tokens: u64,
    pub worker_prompt_tokens: u64,
    pub worker_completion_tokens: u64,
    pub worker_cached_tokens: u64,
    /// Accepted tokens per speculative step, per role, when measurable.
    pub planner_acceptance: Option<f64>,
    pub worker_acceptance: Option<f64>,
    /// Fraction of extracted facts rejected at verify.
    pub rejection_rate: f64,
    /// Facts committed this turn.
    pub facts_committed: usize,
    /// The engine's cache page (or hybrid snapshot granularity) in tokens, when
    /// known. `0` means unknown, and the cache-hit floor is then applied as
    /// written.
    pub page_size: u64,
    /// Estimated tokens of the byte-stable prefix this turn's requests share.
    /// `0` means unknown.
    pub stable_prefix_tokens: u64,
    /// The most tokens the engine *could* have served from cache for the
    /// prompts this turn sent, per role: every full page before the page that
    /// holds the volatile tail. That last page can never hit, so on a
    /// 256-token page a three-page prompt tops out at 67% however stable the
    /// prefix is (measured on DeepSeek-V4.1 under the `dsv4` backend, where
    /// the harness's ~770-token requests cached exactly 512 every time).
    pub planner_achievable_tokens: u64,
    pub worker_achievable_tokens: u64,
}

impl TurnMetrics {
    pub fn new(turn: TurnIndex) -> Self {
        Self {
            turn,
            ..Self::default()
        }
    }

    /// State the engine's page size so the cache-hit floor can be bounded by
    /// what a prompt of that length can achieve at all.
    pub fn with_page_size(mut self, page_size: u32) -> Self {
        self.page_size = page_size as u64;
        self
    }

    /// State how many tokens of the prompt are the byte-stable prefix
    /// (segments 1–4, padded). Only those pages are *guaranteed* to hit; the
    /// volatile tail may or may not, depending on which step the request is
    /// and what the previous step appended. The floor is therefore "the
    /// stable prefix cached", which is exactly what the breach message
    /// claims to detect.
    pub fn with_stable_prefix_tokens(mut self, tokens: u32) -> Self {
        self.stable_prefix_tokens = tokens as u64;
        self
    }

    /// Tokens a prompt of `prompt_tokens` is expected to have served from
    /// cache: the full pages of the stable prefix, never more than the pages
    /// before the one holding the volatile tail.
    ///
    /// The second bound alone over-promised on the first live run. Steps
    /// within a turn share the stable prefix and the rendered subgraph, but
    /// each appends its own tail -- the plan, the answer text, the extract
    /// instruction -- and once a step's private tail crossed a page boundary
    /// the "all pages but the last" ceiling flagged a breach on a prefix that
    /// was byte-identical. Bounding by the stable prefix asks only for what
    /// the harness controls.
    fn achievable(&self, prompt_tokens: u64) -> u64 {
        if self.page_size == 0 {
            return prompt_tokens;
        }
        let by_pages = prompt_tokens.div_ceil(self.page_size).saturating_sub(1) * self.page_size;
        if self.stable_prefix_tokens == 0 {
            return by_pages;
        }
        let stable_pages = (self.stable_prefix_tokens / self.page_size) * self.page_size;
        by_pages.min(stable_pages)
    }

    /// Record one model call's usage against a role.
    pub fn record_usage(
        &mut self,
        role: ModelRole,
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
    ) {
        let achievable = self.achievable(prompt_tokens);
        match role {
            ModelRole::Planner => {
                self.planner_prompt_tokens += prompt_tokens;
                self.planner_completion_tokens += completion_tokens;
                self.planner_cached_tokens += cached_tokens;
                self.planner_achievable_tokens += achievable;
            }
            ModelRole::Worker => {
                self.worker_prompt_tokens += prompt_tokens;
                self.worker_completion_tokens += completion_tokens;
                self.worker_cached_tokens += cached_tokens;
                self.worker_achievable_tokens += achievable;
            }
        }
    }

    /// Prefix-cache hit rate for one model. `None` when it was not called.
    pub fn cache_hit_rate(&self, role: ModelRole) -> Option<f64> {
        let (cached, prompt) = match role {
            ModelRole::Planner => (self.planner_cached_tokens, self.planner_prompt_tokens),
            ModelRole::Worker => (self.worker_cached_tokens, self.worker_prompt_tokens),
        };
        if prompt == 0 {
            return None;
        }
        Some(cached as f64 / prompt as f64)
    }

    /// The best hit rate this turn's prompts could have reached at the
    /// engine's page size. `None` when the role was not called.
    pub fn cache_hit_ceiling(&self, role: ModelRole) -> Option<f64> {
        let (achievable, prompt) = match role {
            ModelRole::Planner => (self.planner_achievable_tokens, self.planner_prompt_tokens),
            ModelRole::Worker => (self.worker_achievable_tokens, self.worker_prompt_tokens),
        };
        if prompt == 0 {
            return None;
        }
        Some(achievable as f64 / prompt as f64)
    }

    /// Total tokens for the turn, the number the "trending down" metric tracks.
    pub fn total_tokens(&self) -> u64 {
        self.planner_prompt_tokens
            + self.planner_completion_tokens
            + self.worker_prompt_tokens
            + self.worker_completion_tokens
    }

    fn planner_tokens(&self) -> u64 {
        self.planner_prompt_tokens + self.planner_completion_tokens
    }

    fn worker_tokens(&self) -> u64 {
        self.worker_prompt_tokens + self.worker_completion_tokens
    }

    /// Planner tokens divided by worker tokens. Spec target: <= 1/3.
    ///
    /// `None` when the worker was not called — a turn with no worker work has
    /// no ratio, and reporting infinity would look like a catastrophic breach.
    pub fn planner_worker_ratio(&self) -> Option<f64> {
        if self.worker_tokens() == 0 {
            return None;
        }
        Some(self.planner_tokens() as f64 / self.worker_tokens() as f64)
    }

    /// Every threshold this turn breached, phrased as a defect.
    pub fn breaches(&self, t: Thresholds) -> Vec<Breach> {
        let mut out = Vec::new();

        for role in [ModelRole::Planner, ModelRole::Worker] {
            let Some(rate) = self.cache_hit_rate(role) else {
                continue;
            };
            // The floor cannot ask for more than the page size allows. A
            // prompt that spans three pages can cache two of them at best;
            // holding it to 85% would report a defect in a prefix that is
            // doing everything a prefix can do. The ceiling is compared with
            // a small tolerance because cached_tokens is page-granular and
            // the ratio lands exactly on it when the prefix is perfect.
            let ceiling = self.cache_hit_ceiling(role).unwrap_or(1.0);
            let floor = t.cache_hit_rate.min(ceiling);
            if rate + 1e-9 < floor {
                let bounded = ceiling < t.cache_hit_rate;
                out.push(Breach {
                    kind: BreachKind::CacheHitRate,
                    detail: if bounded {
                        format!(
                            "{role} prefix-cache hit rate {:.1}% is below the {:.1}% this \
                             turn's prompts could reach at a {}-token page (the spec floor \
                             is {:.0}%). Something changed the stable prefix; check the \
                             segment hashes.",
                            rate * 100.0,
                            ceiling * 100.0,
                            self.page_size,
                            t.cache_hit_rate * 100.0
                        )
                    } else {
                        format!(
                            "{role} prefix-cache hit rate {:.1}% is below the {:.0}% floor. \
                             Something changed the stable prefix; check the segment hashes.",
                            rate * 100.0,
                            t.cache_hit_rate * 100.0
                        )
                    },
                });
            }
        }

        if let Some(a) = self.planner_acceptance
            && a < t.planner_acceptance
        {
            out.push(Breach {
                kind: BreachKind::Acceptance,
                detail: format!(
                    "planner acceptance {a:.2} tokens/step is below the {:.2} floor",
                    t.planner_acceptance
                ),
            });
        }
        if let Some(a) = self.worker_acceptance
            && a < t.worker_acceptance
        {
            out.push(Breach {
                kind: BreachKind::Acceptance,
                detail: format!(
                    "worker acceptance {a:.2} tokens/step is below the {:.2} floor",
                    t.worker_acceptance
                ),
            });
        }

        if self.rejection_rate > t.max_rejection_rate {
            out.push(Breach {
                kind: BreachKind::RejectionRate,
                detail: format!(
                    "verify rejected {:.0}% of extracted facts, over the {:.0}% ceiling. \
                     The extract prompt or schema is producing facts the rules cannot accept.",
                    self.rejection_rate * 100.0,
                    t.max_rejection_rate * 100.0
                ),
            });
        }

        if let Some(max) = t.max_planner_worker_ratio
            && let Some(r) = self.planner_worker_ratio()
            && r > max
        {
            out.push(Breach {
                kind: BreachKind::TokenRatio,
                detail: format!(
                    "planner:worker token ratio {r:.2} exceeds the {max:.2} target; \
                     work that belongs on the worker is running on the planner"
                ),
            });
        }

        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreachKind {
    CacheHitRate,
    Acceptance,
    RejectionRate,
    TokenRatio,
}

/// A threshold breach, phrased as a defect rather than a datapoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breach {
    pub kind: BreachKind,
    pub detail: String,
}

/// Rolling metrics across a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub turns: Vec<TurnMetrics>,
    pub thresholds: Thresholds,
}

impl SessionMetrics {
    pub fn new(thresholds: Thresholds) -> Self {
        Self {
            turns: Vec::new(),
            thresholds,
        }
    }

    pub fn record(&mut self, m: TurnMetrics) {
        self.turns.push(m);
    }

    pub fn latest(&self) -> Option<&TurnMetrics> {
        self.turns.last()
    }

    /// Mean cache hit rate for a role across turns that called it.
    pub fn mean_cache_hit_rate(&self, role: ModelRole) -> Option<f64> {
        let rates: Vec<f64> = self
            .turns
            .iter()
            .filter_map(|t| t.cache_hit_rate(role))
            .collect();
        if rates.is_empty() {
            return None;
        }
        Some(rates.iter().sum::<f64>() / rates.len() as f64)
    }

    /// Whether tokens per turn are trending down, comparing the first and
    /// second halves of the session.
    ///
    /// `None` until there are enough turns to say anything: with two or three
    /// turns the comparison is noise, and reporting it as a trend invites
    /// tuning against randomness.
    pub fn tokens_trending_down(&self) -> Option<bool> {
        const MIN_TURNS: usize = 6;
        if self.turns.len() < MIN_TURNS {
            return None;
        }
        let mid = self.turns.len() / 2;
        let mean = |slice: &[TurnMetrics]| {
            slice.iter().map(|t| t.total_tokens()).sum::<u64>() as f64 / slice.len() as f64
        };
        Some(mean(&self.turns[mid..]) < mean(&self.turns[..mid]))
    }

    /// Every breach in the most recent turn.
    pub fn latest_breaches(&self) -> Vec<Breach> {
        self.latest()
            .map(|t| t.breaches(self.thresholds))
            .unwrap_or_default()
    }

    /// Session-end facts for the self-tuning loop.
    ///
    /// Written as [`Source::Harness`] facts, which
    /// [`qgi2_spec_types::Source::is_trusted`] marks as bypassing verify: they
    /// are measurements the harness took, not claims a model made.
    pub fn to_facts(&self, turn: TurnIndex) -> Vec<Fact> {
        let mut out = Vec::new();
        let mut push = |subject: &str, relation: Relation, object: String| {
            let key = FactKey::new(subject, relation, object);
            out.push(Fact {
                id: key.id(),
                key,
                confidence: Confidence::ONE,
                source: Source::Harness,
                turn,
                reinforcements: 1,
                superseded_by: None,
            });
        };

        for role in [ModelRole::Planner, ModelRole::Worker] {
            if let Some(r) = self.mean_cache_hit_rate(role) {
                push(
                    &format!("metric:cache_hit_rate:{role}"),
                    Relation::IsA,
                    // Two decimals: a metric fact that changes every session by
                    // a rounding artifact would churn the durable slice, which
                    // sits in the cached prefix.
                    format!("{r:.2}"),
                );
            }
        }

        if let Some(t) = self.latest() {
            if let Some(a) = t.planner_acceptance {
                push(
                    "metric:acceptance:planner",
                    Relation::IsA,
                    format!("{a:.2}"),
                );
            }
            if let Some(a) = t.worker_acceptance {
                push("metric:acceptance:worker", Relation::IsA, format!("{a:.2}"));
            }
        }

        if let Some(down) = self.tokens_trending_down() {
            push(
                "metric:tokens_per_turn",
                Relation::IsA,
                if down { "decreasing" } else { "not_decreasing" }.to_string(),
            );
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_turn(turn: TurnIndex) -> TurnMetrics {
        let mut m = TurnMetrics::new(turn);
        m.record_usage(ModelRole::Planner, 1000, 100, 900);
        m.record_usage(ModelRole::Worker, 2000, 2000, 1800);
        m.planner_acceptance = Some(2.0);
        m.worker_acceptance = Some(2.5);
        m.rejection_rate = 0.05;
        m
    }

    #[test]
    fn a_healthy_turn_breaches_nothing() {
        assert!(healthy_turn(1).breaches(Thresholds::default()).is_empty());
    }

    #[test]
    fn a_cache_miss_is_reported_as_a_bug_with_a_next_step() {
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 1000, 100, 100);
        let b = m.breaches(Thresholds::default());
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].kind, BreachKind::CacheHitRate);
        assert!(b[0].detail.contains("segment hashes"), "{}", b[0].detail);
    }

    #[test]
    fn the_cache_floor_is_bounded_by_what_the_page_size_allows() {
        // Measured shape: four ~770-token requests on a 256-token page, each
        // caching exactly the two full pages before the volatile one. 67% is
        // the ceiling for that prompt, not a defect.
        let mut m = TurnMetrics::new(1).with_page_size(256);
        for _ in 0..4 {
            m.record_usage(ModelRole::Planner, 768, 50, 512);
        }
        assert!((m.cache_hit_ceiling(ModelRole::Planner).unwrap() - 2.0 / 3.0).abs() < 1e-9);
        assert!(m.breaches(Thresholds::default()).is_empty());

        // The same prompts caching only one page each IS a defect, and the
        // message names the ceiling rather than the unreachable spec floor.
        let mut m = TurnMetrics::new(1).with_page_size(256);
        m.record_usage(ModelRole::Planner, 768, 50, 256);
        let b = m.breaches(Thresholds::default());
        assert_eq!(b.len(), 1);
        assert!(b[0].detail.contains("66.7%"), "{}", b[0].detail);
        assert!(b[0].detail.contains("256-token page"), "{}", b[0].detail);

        // Unknown page size: the floor applies as written.
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 768, 50, 512);
        assert_eq!(m.breaches(Thresholds::default()).len(), 1);
    }

    #[test]
    fn the_floor_asks_only_for_the_stable_prefix_when_it_is_known() {
        // Stable prefix of 2 pages inside a 4-page prompt: a step whose own
        // tail spans the last two pages can cache exactly 512, and that is
        // not a defect. Without the stable-prefix bound the ceiling would be
        // 768 and this would breach.
        let mut m = TurnMetrics::new(1)
            .with_page_size(256)
            .with_stable_prefix_tokens(540);
        m.record_usage(ModelRole::Planner, 1024, 50, 512);
        assert!(m.breaches(Thresholds::default()).is_empty());

        // Fewer than the stable pages cached IS the defect the message names.
        let mut m = TurnMetrics::new(1)
            .with_page_size(256)
            .with_stable_prefix_tokens(540);
        m.record_usage(ModelRole::Planner, 1024, 50, 256);
        assert_eq!(m.breaches(Thresholds::default()).len(), 1);
    }

    #[test]
    fn a_long_prompt_is_still_held_to_the_spec_floor() {
        // Ten pages, one volatile: the ceiling is 90%, above the 85% floor,
        // so the floor is what applies.
        let mut m = TurnMetrics::new(1).with_page_size(256);
        m.record_usage(ModelRole::Planner, 2560, 50, 2048);
        let b = m.breaches(Thresholds::default());
        assert_eq!(b.len(), 1, "{b:?}");
        assert!(b[0].detail.contains("85% floor"), "{}", b[0].detail);
    }

    #[test]
    fn an_uncalled_model_has_no_hit_rate_and_breaches_nothing() {
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 1000, 10, 950);
        assert_eq!(m.cache_hit_rate(ModelRole::Worker), None);
        assert!(m.breaches(Thresholds::default()).is_empty());
    }

    #[test]
    fn a_turn_with_no_worker_work_has_no_ratio_rather_than_infinity() {
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 1000, 100, 950);
        assert_eq!(m.planner_worker_ratio(), None);
        assert!(
            !m.breaches(Thresholds::default())
                .iter()
                .any(|b| b.kind == BreachKind::TokenRatio)
        );
    }

    #[test]
    fn too_much_planner_work_breaches_the_ratio() {
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 5000, 1000, 4900);
        m.record_usage(ModelRole::Worker, 500, 100, 490);
        let b = m.breaches(Thresholds::default());
        assert!(b.iter().any(|b| b.kind == BreachKind::TokenRatio));
    }

    #[test]
    fn a_high_rejection_rate_points_at_the_extract_step() {
        let mut m = healthy_turn(1);
        m.rejection_rate = 0.4;
        let b = m.breaches(Thresholds::default());
        let r = b
            .iter()
            .find(|b| b.kind == BreachKind::RejectionRate)
            .unwrap();
        assert!(
            r.detail.contains("extract prompt or schema"),
            "{}",
            r.detail
        );
    }

    #[test]
    fn low_acceptance_breaches_per_role() {
        let mut m = healthy_turn(1);
        m.planner_acceptance = Some(1.0);
        m.worker_acceptance = Some(1.2);
        let b = m.breaches(Thresholds::default());
        assert_eq!(
            b.iter()
                .filter(|b| b.kind == BreachKind::Acceptance)
                .count(),
            2
        );
    }

    #[test]
    fn unmeasured_acceptance_is_not_a_breach() {
        let mut m = healthy_turn(1);
        m.planner_acceptance = None;
        m.worker_acceptance = None;
        assert!(m.breaches(Thresholds::default()).is_empty());
    }

    #[test]
    fn a_short_session_reports_no_token_trend() {
        let mut s = SessionMetrics::default();
        for t in 1..=3 {
            s.record(healthy_turn(t));
        }
        assert_eq!(s.tokens_trending_down(), None);
    }

    #[test]
    fn a_falling_token_count_is_detected() {
        let mut s = SessionMetrics::default();
        for t in 1..=4 {
            let mut m = TurnMetrics::new(t);
            m.record_usage(ModelRole::Planner, 5000, 500, 4000);
            s.record(m);
        }
        for t in 5..=8 {
            let mut m = TurnMetrics::new(t);
            m.record_usage(ModelRole::Planner, 1000, 100, 900);
            s.record(m);
        }
        assert_eq!(s.tokens_trending_down(), Some(true));
    }

    #[test]
    fn session_end_writes_metric_facts_the_next_session_can_read() {
        let mut s = SessionMetrics::default();
        for t in 1..=8 {
            s.record(healthy_turn(t));
        }
        let facts = s.to_facts(8);
        assert!(facts.iter().all(|f| f.source == Source::Harness));
        assert!(facts.iter().all(|f| f.source.is_trusted()));
        assert!(
            facts
                .iter()
                .any(|f| f.subject() == "metric:cache_hit_rate:planner")
        );
        assert!(
            facts
                .iter()
                .any(|f| f.subject() == "metric:tokens_per_turn")
        );
    }

    #[test]
    fn metric_facts_are_rounded_so_they_do_not_churn_the_durable_slice() {
        let mut s = SessionMetrics::default();
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 30001, 0, 27123);
        s.record(m);
        let f = s
            .to_facts(1)
            .into_iter()
            .find(|f| f.subject() == "metric:cache_hit_rate:planner")
            .unwrap();
        assert_eq!(
            f.object().len(),
            4,
            "expected two decimals, got {}",
            f.object()
        );
    }

    #[test]
    fn metric_facts_are_deterministic() {
        let mut s = SessionMetrics::default();
        for t in 1..=8 {
            s.record(healthy_turn(t));
        }
        assert_eq!(s.to_facts(8), s.to_facts(8));
    }
}

#[cfg(test)]
mod single_model_tests {
    use super::*;

    #[test]
    fn single_model_thresholds_drop_only_the_ratio() {
        let t = Thresholds::single_model();
        assert!(t.max_planner_worker_ratio.is_none());
        // Everything else is a property of the harness, not of the split.
        assert_eq!(t.cache_hit_rate, Thresholds::default().cache_hit_rate);
        assert_eq!(
            t.max_rejection_rate,
            Thresholds::default().max_rejection_rate
        );
        assert_eq!(t.worker_acceptance, Thresholds::default().worker_acceptance);
    }

    #[test]
    fn a_lopsided_ratio_is_not_a_breach_in_single_model_mode() {
        // With one model there is no cheaper model for work to land on, so the
        // ratio would just describe the step mix and breach on every turn.
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 9000, 1000, 8000);
        m.record_usage(ModelRole::Worker, 500, 100, 400);
        assert!(
            m.breaches(Thresholds::default())
                .iter()
                .any(|b| b.kind == BreachKind::TokenRatio),
            "the two-model default must still catch it"
        );
        assert!(
            !m.breaches(Thresholds::single_model())
                .iter()
                .any(|b| b.kind == BreachKind::TokenRatio)
        );
    }

    #[test]
    fn single_model_mode_still_catches_a_cache_drop() {
        // Suppressing the ratio must not quietly disarm the metric the spec
        // actually calls a bug.
        let mut m = TurnMetrics::new(1);
        m.record_usage(ModelRole::Planner, 1000, 100, 100);
        let b = m.breaches(Thresholds::single_model());
        assert!(b.iter().any(|b| b.kind == BreachKind::CacheHitRate));
    }
}
