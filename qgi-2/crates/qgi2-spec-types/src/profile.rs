//! Profiles: how carefully the agent works. Orthogonal to [`crate::Mood`].
//!
//! Transcribed from the spec's profile table:
//!
//! | | Traceable | Deterministic | Quick |
//! |---|---|---|---|
//! | Worker spec | DFlash2 n=7 | **MTP n=3** (DFlash2 can't do greedy) | DFlash2 n=7, n-gram fallback |
//! | Sampling | as mood | T 0, seed fixed, batch-invariant | as mood, thinking off |
//! | Memory sync | async (jcode default) | **sync** (await side-agent) | async |
//! | Retrieval | full chain + reranker | full chain, turn-based decay | exact-key + lexical, BFS depth 1 |
//! | Logging | prompts, segment hashes, rule firings, acceptance rates | hashes + seeds + engine build | errors only |
//!
//! The Deterministic row is the load-bearing one: DFlash2 cannot produce greedy
//! output, so a profile that promises `T 0` must switch the worker to MTP. That
//! coupling lives in [`Profile::worker_speculation`] rather than in the router,
//! so it cannot be forgotten at a call site.

use crate::step::{Sampling, Speculation};
use serde::{Deserialize, Serialize};
use std::fmt;

/// How carefully the agent works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// Full logging, full retrieval, speculation on.
    Traceable,
    /// Reproducible: greedy sampling, fixed seed, sync memory.
    Deterministic,
    /// Latency first: shallow retrieval, thinking off.
    Quick,
}

impl Profile {
    pub const ALL: [Profile; 3] = [Profile::Traceable, Profile::Deterministic, Profile::Quick];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Traceable => "traceable",
            Self::Deterministic => "deterministic",
            Self::Quick => "quick",
        }
    }

    /// The worker's speculation method under this profile.
    ///
    /// Deterministic returns MTP, not DFlash2: DFlash2 cannot do greedy
    /// decoding, and the spec's "speculation never changes output
    /// distribution" invariant means a `T 0` profile cannot use it.
    pub const fn worker_speculation(self) -> Speculation {
        match self {
            Self::Traceable => Speculation::DFlash2 { n: 7 },
            Self::Deterministic => Speculation::Mtp { n: 3 },
            Self::Quick => Speculation::DFlash2 { n: 7 },
        }
    }

    /// Speculation to fall back to when the primary method is unavailable for
    /// a step (e.g. output that mostly copies the prompt).
    pub const fn worker_speculation_fallback(self) -> Option<Speculation> {
        match self {
            Self::Quick => Some(Speculation::NGram { n: 4 }),
            // Traceable wants the measured acceptance of one method; Deterministic
            // cannot swap methods without risking a different distribution.
            Self::Traceable | Self::Deterministic => None,
        }
    }

    /// Apply the profile's sampling override to a mood-derived sampling.
    ///
    /// Traceable takes the mood's sampling unchanged; Deterministic forces
    /// greedy with a fixed seed and batch-invariant kernels; Quick keeps the
    /// mood's temperature but disables thinking.
    pub fn apply_sampling(self, mood_sampling: Sampling) -> Sampling {
        match self {
            Self::Traceable => mood_sampling,
            Self::Deterministic => Sampling {
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(DETERMINISTIC_SEED),
                batch_invariant: true,
                thinking: mood_sampling.thinking,
                max_tokens: mood_sampling.max_tokens,
            },
            Self::Quick => Sampling {
                thinking: false,
                ..mood_sampling
            },
        }
    }

    pub const fn memory_sync(self) -> MemorySync {
        match self {
            Self::Deterministic => MemorySync::Sync,
            Self::Traceable | Self::Quick => MemorySync::Async,
        }
    }

    pub const fn retrieval(self) -> RetrievalPolicy {
        match self {
            Self::Traceable => RetrievalPolicy {
                full_chain: true,
                reranker: true,
                turn_based_decay: false,
                lexical_only: false,
                max_depth: u8::MAX,
            },
            Self::Deterministic => RetrievalPolicy {
                full_chain: true,
                // A reranker is a second model call whose output is not pinned
                // by the seed, so Deterministic cannot use one.
                reranker: false,
                turn_based_decay: true,
                lexical_only: false,
                max_depth: u8::MAX,
            },
            Self::Quick => RetrievalPolicy {
                full_chain: false,
                reranker: false,
                turn_based_decay: false,
                lexical_only: true,
                max_depth: 1,
            },
        }
    }

    pub const fn logging(self) -> LoggingPolicy {
        match self {
            Self::Traceable => LoggingPolicy {
                prompts: true,
                segment_hashes: true,
                rule_firings: true,
                acceptance_rates: true,
                seeds: false,
                engine_build: false,
                errors_only: false,
            },
            Self::Deterministic => LoggingPolicy {
                prompts: false,
                segment_hashes: true,
                rule_firings: false,
                acceptance_rates: false,
                seeds: true,
                engine_build: true,
                errors_only: false,
            },
            Self::Quick => LoggingPolicy {
                prompts: false,
                segment_hashes: false,
                rule_firings: false,
                acceptance_rates: false,
                seeds: false,
                engine_build: false,
                errors_only: true,
            },
        }
    }

    /// One row of the profile table, for rendering and inspection.
    pub fn table(self) -> ProfileTable {
        ProfileTable {
            profile: self,
            worker_speculation: self.worker_speculation(),
            worker_speculation_fallback: self.worker_speculation_fallback(),
            memory_sync: self.memory_sync(),
            retrieval: self.retrieval(),
            logging: self.logging(),
        }
    }
}

/// The fixed seed used by the Deterministic profile.
///
/// A constant rather than a random-per-session value: reproducing a run across
/// machines is the point of the profile, and a per-session seed would only be
/// reproducible with the session log.
pub const DETERMINISTIC_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "traceable" => Ok(Self::Traceable),
            "deterministic" => Ok(Self::Deterministic),
            "quick" => Ok(Self::Quick),
            other => Err(format!(
                "unknown profile {other:?}; expected traceable, deterministic, or quick"
            )),
        }
    }
}

/// Whether memory extraction blocks the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySync {
    /// jcode's default: extraction runs in the background.
    Async,
    /// Await the side-agent before answering, so the same input produces the
    /// same graph state at answer time.
    Sync,
}

impl MemorySync {
    pub const fn is_sync(self) -> bool {
        matches!(self, Self::Sync)
    }
}

/// How much of the graph retrieval walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalPolicy {
    /// Walk the full mood traversal chain rather than stopping at entry points.
    pub full_chain: bool,
    /// Run a reranker over retrieved entries.
    pub reranker: bool,
    /// Weight entries by how recently they were reinforced, in turns.
    pub turn_based_decay: bool,
    /// Skip embeddings; use exact-key and lexical matching only.
    pub lexical_only: bool,
    /// Maximum BFS depth from the entry points.
    pub max_depth: u8,
}

/// What gets written to the session log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingPolicy {
    pub prompts: bool,
    pub segment_hashes: bool,
    pub rule_firings: bool,
    pub acceptance_rates: bool,
    pub seeds: bool,
    pub engine_build: bool,
    pub errors_only: bool,
}

/// One row of the profile table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileTable {
    pub profile: Profile,
    pub worker_speculation: Speculation,
    pub worker_speculation_fallback: Option<Speculation>,
    pub memory_sync: MemorySync,
    pub retrieval: RetrievalPolicy,
    pub logging: LoggingPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_uses_mtp_because_dflash2_cannot_do_greedy() {
        assert_eq!(
            Profile::Deterministic.worker_speculation(),
            Speculation::Mtp { n: 3 }
        );
        let s = Profile::Deterministic.apply_sampling(Sampling::at_temperature(0.7));
        assert_eq!(s.temperature, 0.0);
        assert_eq!(s.seed, Some(DETERMINISTIC_SEED));
        assert!(s.batch_invariant);
    }

    #[test]
    fn greedy_sampling_never_pairs_with_a_non_greedy_speculator() {
        // The spec's invariant is that speculation must not change the output
        // distribution. Any profile that forces T=0 must therefore use a
        // speculator that supports greedy.
        for p in Profile::ALL {
            let s = p.apply_sampling(Sampling::at_temperature(0.7));
            if s.temperature == 0.0 {
                assert!(
                    p.worker_speculation().supports_greedy(),
                    "{p} forces greedy but uses {:?}",
                    p.worker_speculation()
                );
            }
        }
    }

    #[test]
    fn traceable_passes_mood_sampling_through_unchanged() {
        let mood = Sampling::at_temperature(0.3);
        assert_eq!(Profile::Traceable.apply_sampling(mood), mood);
    }

    #[test]
    fn quick_keeps_temperature_but_drops_thinking() {
        let mood = Sampling {
            thinking: true,
            ..Sampling::at_temperature(0.7)
        };
        let s = Profile::Quick.apply_sampling(mood);
        assert_eq!(s.temperature, 0.7);
        assert!(!s.thinking);
    }

    #[test]
    fn quick_retrieval_is_depth_one_and_lexical() {
        let r = Profile::Quick.retrieval();
        assert_eq!(r.max_depth, 1);
        assert!(r.lexical_only);
        assert!(!r.full_chain);
    }

    #[test]
    fn only_deterministic_blocks_on_memory() {
        assert!(Profile::Deterministic.memory_sync().is_sync());
        assert!(!Profile::Traceable.memory_sync().is_sync());
        assert!(!Profile::Quick.memory_sync().is_sync());
    }

    #[test]
    fn deterministic_does_not_rerank() {
        // A reranker is an unpinned second model call; it would break
        // reproducibility even with the sampling seed fixed.
        assert!(!Profile::Deterministic.retrieval().reranker);
    }

    #[test]
    fn profiles_round_trip_through_strings() {
        for p in Profile::ALL {
            assert_eq!(p.as_str().parse::<Profile>().unwrap(), p);
        }
    }
}
