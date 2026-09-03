//! The seam between the turn loop and whatever actually runs tools.
//!
//! QGI-2 is driven by an outer agent loop it does not own — jcode's, in both
//! edges. That loop executes tools itself and re-enters the provider with the
//! results, so the common case is that QGI-2 *emits* a call and never runs it.
//!
//! [`ToolDisposition`] is what makes that explicit. A runner either executes a
//! call or declines it, and declining is not a failure: it means "the caller
//! will run this, hand it back to them". An earlier version of this trait had
//! `run` return a canned "deferred" string, which the loop then fed into fact
//! extraction as though it were a tool result — the disposition type exists so
//! that cannot happen again.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's arguments. Spliced into the tool-args step's
    /// guided-decoding schema so a malformed call is impossible rather than
    /// merely unlikely.
    pub parameters: Value,
}

/// A call the model asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Stable id, so a result can be matched back to its call across the
    /// round trip through the outer agent loop.
    pub id: String,
    pub tool: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Deterministic id from the round, position, and tool name.
    ///
    /// Deterministic rather than random so a Deterministic-profile run produces
    /// the same ids on every replay; a UUID here would be the one piece of a
    /// reproducible turn that was not reproducible.
    pub fn id_for(round: u32, index: usize, tool: &str) -> String {
        format!("qgi2-{round}-{index}-{tool}")
    }
}

/// What running a call produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub call: ToolCall,
    pub output: String,
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn ok(call: ToolCall, output: impl Into<String>) -> Self {
        Self {
            call,
            output: output.into(),
            is_error: false,
        }
    }

    pub fn error(call: ToolCall, message: impl Into<String>) -> Self {
        Self {
            call,
            output: message.into(),
            is_error: true,
        }
    }

    /// Render for the model, tagged so the answer step can tell a failure from
    /// a successful call that happened to print the word "error".
    pub fn render(&self) -> String {
        self.render_capped(usize::MAX)
    }

    /// Render with the output windowed to at most `limit` bytes.
    ///
    /// A single `read` of a large file otherwise lands in full in the extract
    /// prompt *and* the answer prompt -- twice per round, on the two most
    /// expensive steps -- when the model needs the shape of it, not every byte.
    /// The head carries most of a file's structure (imports, signatures) and the
    /// tail its most recent state, so the window keeps three quarters from the
    /// front and one from the back, and says exactly how much it dropped.
    pub fn render_capped(&self, limit: usize) -> String {
        let body = if self.output.len() <= limit {
            self.output.clone()
        } else {
            let head_len = floor_char_boundary(&self.output, limit * 3 / 4);
            let tail_start = ceil_char_boundary(&self.output, self.output.len() - limit / 4);
            let omitted = tail_start - head_len;
            format!(
                "{}\n\u{2026} [{omitted} bytes omitted] \u{2026}\n{}",
                &self.output[..head_len],
                &self.output[tail_start..]
            )
        };
        if self.is_error {
            format!("{} [FAILED] -> {}", self.call.tool, body)
        } else {
            format!("{} -> {}", self.call.tool, body)
        }
    }

    /// Whether `render_capped(limit)` would drop anything.
    pub fn exceeds(&self, limit: usize) -> bool {
        self.output.len() > limit
    }
}

/// Largest index `<= i` that is a char boundary. (`str::floor_char_boundary`
/// is unstable at this toolchain.)
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest index `>= i` that is a char boundary.
fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Whether a runner executed a call or left it to the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDisposition {
    /// The runner ran it.
    Executed(ToolOutcome),
    /// The caller's agent loop will run it; the loop returns it upward.
    Deferred(ToolCall),
}

/// Supplies the tool catalogue and, optionally, executes calls.
#[async_trait]
pub trait ToolRunner: Send + Sync {
    /// Every tool available, before the mood mask is applied.
    async fn available(&self) -> Result<Vec<ToolSpec>>;

    /// Execute a call, or decline so the caller executes it.
    async fn run(&self, call: ToolCall) -> Result<ToolDisposition>;
}

/// Reports a catalogue and defers every call to the caller's agent loop.
///
/// Both edges use this: jcode owns tool execution, and QGI-2's job is to decide
/// *which* tool with *which* arguments, under the mood mask and a constrained
/// schema.
pub struct DeferToCaller {
    specs: Vec<ToolSpec>,
}

impl DeferToCaller {
    pub fn new(specs: Vec<ToolSpec>) -> Self {
        Self { specs }
    }
}

#[async_trait]
impl ToolRunner for DeferToCaller {
    async fn available(&self) -> Result<Vec<ToolSpec>> {
        Ok(self.specs.clone())
    }

    async fn run(&self, call: ToolCall) -> Result<ToolDisposition> {
        Ok(ToolDisposition::Deferred(call))
    }
}

/// A runner with no tools at all.
///
/// `available` is empty, so the mood mask admits nothing and the loop never
/// reaches `run`. Used when a caller wants QGI-2 as a pure reasoning endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTools;

#[async_trait]
impl ToolRunner for NoTools {
    async fn available(&self) -> Result<Vec<ToolSpec>> {
        Ok(Vec::new())
    }

    async fn run(&self, call: ToolCall) -> Result<ToolDisposition> {
        Ok(ToolDisposition::Executed(ToolOutcome::error(
            call,
            "this harness has no tools configured",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call() -> ToolCall {
        ToolCall {
            id: ToolCall::id_for(0, 0, "read"),
            tool: "read".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        }
    }

    #[tokio::test]
    async fn the_deferring_runner_hands_the_call_back_unchanged() {
        let r = DeferToCaller::new(vec![]);
        match r.run(call()).await.unwrap() {
            ToolDisposition::Deferred(c) => assert_eq!(c, call()),
            other => panic!("expected Deferred, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deferral_is_never_mistaken_for_a_tool_result() {
        // The bug this type prevents: a canned "deferred" string being fed to
        // fact extraction as though a tool had actually run.
        let r = DeferToCaller::new(vec![]);
        let d = r.run(call()).await.unwrap();
        assert!(!matches!(d, ToolDisposition::Executed(_)));
    }

    #[test]
    fn call_ids_are_deterministic() {
        assert_eq!(ToolCall::id_for(1, 2, "edit"), ToolCall::id_for(1, 2, "edit"));
        assert_ne!(ToolCall::id_for(1, 2, "edit"), ToolCall::id_for(1, 3, "edit"));
    }

    #[test]
    fn oversized_output_is_windowed_head_and_tail() {
        let big = format!("{}MIDDLE{}", "a".repeat(600), "z".repeat(600));
        let o = ToolOutcome::ok(call(), big);
        let r = o.render_capped(400);
        assert!(r.len() < 600, "len {}", r.len());
        assert!(r.contains("bytes omitted"));
        assert!(r.contains(&"a".repeat(100)), "head kept");
        assert!(r.contains(&"z".repeat(50)), "tail kept");
        assert!(!r.contains("MIDDLE"), "middle dropped");
        assert!(o.exceeds(400));
        assert!(!o.exceeds(10_000));
    }

    #[test]
    fn windowing_never_splits_a_multibyte_character() {
        // Every byte of this is inside a 3-byte char; any cut that is not on a
        // boundary would panic at the slice.
        let big = "\u{20ac}".repeat(500);
        let o = ToolOutcome::ok(call(), big);
        let r = o.render_capped(101);
        assert!(r.contains("bytes omitted"));
        assert!(r.chars().all(|c| c != '\u{fffd}'));
    }

    #[test]
    fn output_within_the_limit_is_untouched() {
        let o = ToolOutcome::ok(call(), "short");
        assert_eq!(o.render_capped(1000), o.render());
    }

    #[test]
    fn failed_outcomes_are_tagged_so_the_model_can_tell() {
        let ok = ToolOutcome::ok(call(), "contents");
        let bad = ToolOutcome::error(call(), "no such file");
        assert_eq!(ok.render(), "read -> contents");
        assert!(bad.render().contains("[FAILED]"));
    }

    #[tokio::test]
    async fn the_empty_runner_reports_no_tools() {
        assert!(NoTools.available().await.unwrap().is_empty());
    }
}
