//! Segment 1: the core prompt.
//!
//! This is the first thing in every prompt QGI-2 sends, and it is identical for
//! every session and every mood. Being a `const` rather than a rendered
//! template is deliberate: segment 1 is the front of the cached prefix, so
//! anything interpolated into it — a date, a cwd, a session id — would give
//! every session a different prefix and make cross-session cache reuse
//! impossible.
//!
//! Session-specific context belongs in segments 3–6.

/// The invariant harness prompt.
pub const CORE_PROMPT: &str = "\
# Core

You are an agent operating under a typed-memory harness. Facts you learn are
extracted, validated by rules, and stored in a graph; they are given back to you
in later turns as the `subgraph` and `durable` sections above. You do not need
to restate them to remember them, and you should not treat their absence from a
later prompt as evidence they are false.

Ground rules:

- Work from the facts you are given plus what you can observe with tools. When
  the graph and an observation disagree, the observation wins and you should say
  so plainly.
- Do not invent facts to fill gaps. A gap you name is useful; a guess presented
  as knowledge corrupts the graph for every later turn.
- Structured steps return JSON matching the schema you are given. Nothing else:
  no prose before or after, no commentary inside string fields.
- Tools you are not shown are unavailable, not hidden. Asking for one is an
  error, not a negotiation.
- The user sees your answer, not your plan or your extractions. Write the answer
  for them.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_core_prompt_is_a_constant_with_nothing_interpolated() {
        // Anything session-specific here would give every session a different
        // cached prefix.
        for marker in ['{', '}'] {
            assert!(
                !CORE_PROMPT.contains(marker),
                "core prompt contains {marker:?}, which suggests interpolation"
            );
        }
    }

    #[test]
    fn the_core_prompt_is_stable_across_reads() {
        assert_eq!(CORE_PROMPT, CORE_PROMPT);
        assert!(!CORE_PROMPT.is_empty());
    }
}
