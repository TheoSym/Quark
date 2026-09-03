//! Speculation acceptance, normalised across engines.
//!
//! Spec success metrics: planner MTP >= 1.8 tokens/step, worker DFlash2 >= 2.0.
//!
//! Neither engine reports per-request acceptance on the OpenAI-compatible API,
//! so both are scraped from Prometheus. They report it very differently:
//!
//! - **vLLM** exposes monotonic counters (`vllm:spec_decode_num_*`). A single
//!   scrape describes the server's whole lifetime, so the number only becomes
//!   about *this turn* by subtracting an earlier scrape — see
//!   [`AcceptanceSnapshot::since`].
//! - **SGLang** exposes `sglang:spec_accept_length` as a gauge that already is
//!   the mean accepted length. There is nothing to subtract, and subtracting
//!   would be wrong.
//!
//! [`AcceptanceSnapshot`] carries both shapes and [`AcceptanceSnapshot::since`]
//! knows which one it is holding, so callers can difference every snapshot
//! uniformly without corrupting the gauge case.

use serde::{Deserialize, Serialize};

/// Acceptance as one engine reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum AcceptanceSnapshot {
    /// Monotonic counters (vLLM). Difference two to get a window.
    Counters {
        draft_tokens: u64,
        accepted_tokens: u64,
        num_steps: u64,
    },
    /// A directly reported mean accepted length (SGLang).
    Gauge { accept_length: f64 },
    /// The engine reported nothing recognizable.
    #[default]
    Unavailable,
}

impl AcceptanceSnapshot {
    /// Mean accepted tokens per speculative step.
    ///
    /// `None` when nothing was measured: reporting 0.0 would look like a breach
    /// of the acceptance floor rather than an absence of data.
    pub fn tokens_per_step(&self) -> Option<f64> {
        match self {
            Self::Counters {
                accepted_tokens,
                num_steps,
                ..
            } => {
                if *num_steps == 0 {
                    return None;
                }
                // +1 because a speculative step always emits the target model's
                // own token in addition to accepted draft tokens. The spec's
                // floors are stated on that basis: 1.0 would mean nothing was
                // ever accepted.
                Some(*accepted_tokens as f64 / *num_steps as f64 + 1.0)
            }
            // SGLang's accept_length already counts the verified token.
            Self::Gauge { accept_length } => {
                (*accept_length > 0.0).then_some(*accept_length)
            }
            Self::Unavailable => None,
        }
    }

    /// Fraction of drafted tokens accepted, where the engine reports drafts.
    pub fn acceptance_ratio(&self) -> Option<f64> {
        match self {
            Self::Counters {
                draft_tokens,
                accepted_tokens,
                ..
            } if *draft_tokens > 0 => Some(*accepted_tokens as f64 / *draft_tokens as f64),
            _ => None,
        }
    }

    /// The window since an earlier snapshot.
    ///
    /// Only meaningful for counters. A gauge is already an instantaneous mean,
    /// so differencing it would produce a meaningless number — this returns the
    /// gauge unchanged instead.
    ///
    /// Counter subtraction saturates: a server restart resets the counters, and
    /// wrapping would report an absurd acceptance rate rather than an empty
    /// window.
    pub fn since(&self, earlier: &Self) -> Self {
        match (self, earlier) {
            (
                Self::Counters {
                    draft_tokens: d1,
                    accepted_tokens: a1,
                    num_steps: s1,
                },
                Self::Counters {
                    draft_tokens: d0,
                    accepted_tokens: a0,
                    num_steps: s0,
                },
            ) => Self::Counters {
                draft_tokens: d1.saturating_sub(*d0),
                accepted_tokens: a1.saturating_sub(*a0),
                num_steps: s1.saturating_sub(*s0),
            },
            (current, _) => *current,
        }
    }

    /// Whether this window clears a floor.
    pub fn meets_floor(&self, floor: f64) -> Option<bool> {
        self.tokens_per_step().map(|t| t >= floor)
    }

    pub fn is_available(&self) -> bool {
        !matches!(self, Self::Unavailable)
    }
}

/// Parse a Prometheus exposition line into `(metric name, value)`.
///
/// Tolerant by design: metric names move between engine versions, and a
/// deployment that reports nothing recognizable should yield "no data" rather
/// than failing a turn.
pub fn prometheus_lines(body: &str) -> impl Iterator<Item = (&str, f64)> {
    body.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (name_part, value_part) = line.rsplit_once(' ')?;
        let name = name_part.split('{').next().unwrap_or(name_part).trim();
        let value: f64 = value_part.trim().parse().ok()?;
        value.is_finite().then_some((name, value))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_acceptance_counts_the_targets_own_token() {
        let s = AcceptanceSnapshot::Counters {
            draft_tokens: 7000,
            accepted_tokens: 5000,
            num_steps: 1000,
        };
        assert_eq!(s.tokens_per_step(), Some(6.0));
        assert_eq!(s.acceptance_ratio(), Some(5000.0 / 7000.0));
    }

    #[test]
    fn a_gauge_is_already_the_answer() {
        let s = AcceptanceSnapshot::Gauge { accept_length: 2.4 };
        assert_eq!(s.tokens_per_step(), Some(2.4));
    }

    #[test]
    fn differencing_a_gauge_leaves_it_alone() {
        // SGLang's accept_length is an instantaneous mean; subtracting an
        // earlier reading would produce a meaningless number.
        let a = AcceptanceSnapshot::Gauge { accept_length: 2.0 };
        let b = AcceptanceSnapshot::Gauge { accept_length: 2.4 };
        assert_eq!(b.since(&a), b);
    }

    #[test]
    fn differencing_counters_gives_the_window() {
        let a = AcceptanceSnapshot::Counters {
            draft_tokens: 7000,
            accepted_tokens: 5000,
            num_steps: 1000,
        };
        let b = AcceptanceSnapshot::Counters {
            draft_tokens: 7700,
            accepted_tokens: 5100,
            num_steps: 1100,
        };
        assert_eq!(b.since(&a).tokens_per_step(), Some(2.0));
    }

    #[test]
    fn a_restart_does_not_produce_an_absurd_rate() {
        let before = AcceptanceSnapshot::Counters {
            draft_tokens: 7000,
            accepted_tokens: 5000,
            num_steps: 1000,
        };
        let after = AcceptanceSnapshot::Counters {
            draft_tokens: 10,
            accepted_tokens: 5,
            num_steps: 2,
        };
        assert_eq!(after.since(&before).tokens_per_step(), None);
    }

    #[test]
    fn no_data_is_not_a_breach() {
        assert_eq!(AcceptanceSnapshot::Unavailable.tokens_per_step(), None);
        assert_eq!(AcceptanceSnapshot::Unavailable.meets_floor(2.0), None);
        assert_eq!(
            AcceptanceSnapshot::Counters {
                draft_tokens: 0,
                accepted_tokens: 0,
                num_steps: 0
            }
            .tokens_per_step(),
            None
        );
        assert_eq!(
            AcceptanceSnapshot::Gauge { accept_length: 0.0 }.tokens_per_step(),
            None
        );
    }

    #[test]
    fn prometheus_parsing_skips_comments_and_junk() {
        let body = "# HELP x\n# TYPE x counter\nfoo{a=\"b\"} 1.5\nbar 2\nnot a metric\n";
        let got: Vec<_> = prometheus_lines(body).collect();
        assert_eq!(got, vec![("foo", 1.5), ("bar", 2.0)]);
    }
}
