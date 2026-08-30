//! Speculation acceptance, scraped from vLLM's Prometheus endpoint.
//!
//! Spec success metrics:
//!
//! > Speculation acceptance: planner MTP >= 1.8 tokens/step; worker DFlash2 >= 2.0.
//!
//! vLLM does not report per-request acceptance on the OpenAI-compatible API, so
//! this reads the process-level counters from `/metrics` instead. They are
//! monotonic totals, which means a single scrape describes the server's whole
//! lifetime and is nearly useless for "was this turn healthy?". Taking the
//! delta between two scrapes is what makes the number about the window you care
//! about — see [`AcceptanceSnapshot::since`].

use crate::client::VllmClient;
use crate::registry::Endpoint;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Counter values at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceSnapshot {
    /// Tokens the draft model proposed.
    pub draft_tokens: u64,
    /// Tokens the target model accepted.
    pub accepted_tokens: u64,
    /// Number of speculative decoding steps.
    pub num_steps: u64,
}

impl AcceptanceSnapshot {
    /// Mean accepted tokens per step over the server's lifetime.
    ///
    /// `None` when no speculative step has run: reporting 0.0 would look like a
    /// breach of the acceptance floor rather than an absence of data.
    pub fn tokens_per_step(&self) -> Option<f64> {
        if self.num_steps == 0 {
            return None;
        }
        // +1 because a speculative step always emits the target model's own
        // token in addition to whatever draft tokens were accepted. The spec's
        // floors (1.8, 2.0) are stated on that basis: 1.0 would mean nothing
        // was ever accepted.
        Some(self.accepted_tokens as f64 / self.num_steps as f64 + 1.0)
    }

    /// Fraction of drafted tokens that were accepted.
    pub fn acceptance_ratio(&self) -> Option<f64> {
        if self.draft_tokens == 0 {
            return None;
        }
        Some(self.accepted_tokens as f64 / self.draft_tokens as f64)
    }

    /// The delta against an earlier snapshot.
    ///
    /// The counters are monotonic totals, so a lifetime reading cannot tell you
    /// whether the last few turns were healthy. Subtracting an earlier scrape
    /// scopes the numbers to the window between them.
    ///
    /// Saturating subtraction: a vLLM restart resets the counters, and a
    /// wrapping subtraction there would report an absurd acceptance rate rather
    /// than a small or empty window.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            draft_tokens: self.draft_tokens.saturating_sub(earlier.draft_tokens),
            accepted_tokens: self.accepted_tokens.saturating_sub(earlier.accepted_tokens),
            num_steps: self.num_steps.saturating_sub(earlier.num_steps),
        }
    }

    /// Whether this window clears the floor for a role's speculation method.
    pub fn meets_floor(&self, floor: f64) -> Option<bool> {
        self.tokens_per_step().map(|t| t >= floor)
    }
}

/// Scrape `/metrics` from a vLLM process.
pub async fn scrape_acceptance(
    client: &VllmClient,
    endpoint: &Endpoint,
) -> Result<AcceptanceSnapshot> {
    // /metrics sits at the server root, not under /v1.
    let root = endpoint
        .base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let url = format!("{root}/metrics");

    let body = client
        .http()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .text()
        .await
        .context("reading metrics body")?;

    Ok(parse_acceptance(&body))
}

/// Parse the speculative-decoding counters out of Prometheus text format.
///
/// Tolerant by design: metric names have moved between vLLM versions, and a
/// deployment that reports nothing recognizable should yield a zero snapshot
/// (which reads as "no data") rather than failing a turn.
pub fn parse_acceptance(body: &str) -> AcceptanceSnapshot {
    let mut snap = AcceptanceSnapshot::default();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `name{labels} value` or `name value`
        let (name_part, value_part) = match line.rsplit_once(' ') {
            Some(p) => p,
            None => continue,
        };
        let name = name_part.split('{').next().unwrap_or(name_part).trim();
        let Ok(value) = value_part.trim().parse::<f64>() else {
            continue;
        };
        let value = if value.is_finite() && value >= 0.0 {
            value as u64
        } else {
            continue;
        };

        match name {
            "vllm:spec_decode_num_draft_tokens_total" | "vllm:spec_decode_num_draft_tokens" => {
                snap.draft_tokens = snap.draft_tokens.max(value);
            }
            "vllm:spec_decode_num_accepted_tokens_total"
            | "vllm:spec_decode_num_accepted_tokens" => {
                snap.accepted_tokens = snap.accepted_tokens.max(value);
            }
            "vllm:spec_decode_num_drafts_total" | "vllm:spec_decode_num_drafts" => {
                snap.num_steps = snap.num_steps.max(value);
            }
            _ => {}
        }
    }

    snap
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{model_name="worker"} 7000.0
vllm:spec_decode_num_accepted_tokens_total{model_name="worker"} 5000.0
vllm:spec_decode_num_drafts_total{model_name="worker"} 1000.0
vllm:num_requests_running{model_name="worker"} 2.0
"#;

    #[test]
    fn counters_parse_out_of_prometheus_text() {
        let s = parse_acceptance(SAMPLE);
        assert_eq!(s.draft_tokens, 7000);
        assert_eq!(s.accepted_tokens, 5000);
        assert_eq!(s.num_steps, 1000);
    }

    #[test]
    fn tokens_per_step_counts_the_targets_own_token() {
        // 5000 accepted over 1000 steps, plus the target's token each step.
        let s = parse_acceptance(SAMPLE);
        assert_eq!(s.tokens_per_step(), Some(6.0));
        assert_eq!(s.acceptance_ratio(), Some(5000.0 / 7000.0));
    }

    #[test]
    fn no_speculative_steps_reads_as_no_data_not_a_breach() {
        let s = AcceptanceSnapshot::default();
        assert_eq!(s.tokens_per_step(), None);
        assert_eq!(s.meets_floor(2.0), None);
    }

    #[test]
    fn a_window_is_the_delta_between_two_scrapes() {
        let a = AcceptanceSnapshot {
            draft_tokens: 7000,
            accepted_tokens: 5000,
            num_steps: 1000,
        };
        let b = AcceptanceSnapshot {
            draft_tokens: 7700,
            accepted_tokens: 5100,
            num_steps: 1100,
        };
        let w = b.since(&a);
        assert_eq!(w.num_steps, 100);
        assert_eq!(w.accepted_tokens, 100);
        assert_eq!(w.tokens_per_step(), Some(2.0));
    }

    #[test]
    fn a_server_restart_does_not_produce_an_absurd_rate() {
        // Counters reset to zero; wrapping subtraction would report a huge
        // acceptance rate instead of an empty window.
        let before = AcceptanceSnapshot {
            draft_tokens: 7000,
            accepted_tokens: 5000,
            num_steps: 1000,
        };
        let after = AcceptanceSnapshot {
            draft_tokens: 10,
            accepted_tokens: 5,
            num_steps: 2,
        };
        let w = after.since(&before);
        assert_eq!(w, AcceptanceSnapshot::default());
        assert_eq!(w.tokens_per_step(), None);
    }

    #[test]
    fn unrecognized_output_yields_no_data_rather_than_an_error() {
        // Metric names have moved between vLLM versions; a turn should not fail
        // because acceptance could not be measured.
        let s = parse_acceptance("some_other_metric 1.0\ngarbage line\n");
        assert_eq!(s, AcceptanceSnapshot::default());
    }

    #[test]
    fn the_floor_check_matches_the_spec_metrics() {
        let planner = AcceptanceSnapshot {
            draft_tokens: 200,
            accepted_tokens: 100,
            num_steps: 100,
        };
        // 100/100 + 1 = 2.0, which clears the planner's 1.8 floor.
        assert_eq!(planner.meets_floor(1.8), Some(true));
        // ...and exactly meets the worker's 2.0.
        assert_eq!(planner.meets_floor(2.0), Some(true));
        assert_eq!(planner.meets_floor(2.5), Some(false));
    }
}
