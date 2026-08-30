//! The per-turn loop.
//!
//! Spec:
//!
//! ```text
//! assemble(core, mood, durable, skills, subgraph, query)   # hashes per segment, cache check
//! → plan        [planner, MTP n=2, thinking per profile]
//! → tool calls  [args under schema; mask from rules]
//! → extract     [worker, DFlash2 n=7 | MTP n=3, schema]
//! → verify      [rules: dedupe, conflict, confidence floor]
//! → commit      [graph write, derived views refresh]
//! → answer      [planner]
//! → extract answer facts → verify → commit
//! → mood check  [rules]
//! ```
//!
//! Session end: promote to durable, decay, log speculation acceptance and
//! cache-hit stats as facts for the self-tuning loop.
//!
//! Everything the loop needs from the outside — running a tool, listing which
//! tools exist — arrives through [`ToolRunner`], so the same loop serves both
//! edges: the HTTP sidecar (tools run in the caller) and the in-process jcode
//! provider (tools run in jcode's registry).

pub mod session;
pub mod steps;
pub mod tools;

pub use session::{RoundInput, RoundOutcome, Session, SessionConfig, SessionEnd, TurnResult};
pub use tools::{DeferToCaller, NoTools, ToolCall, ToolDisposition, ToolOutcome, ToolRunner, ToolSpec};
