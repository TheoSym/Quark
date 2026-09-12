//! Line-store memory: the Q-hypergraph design ported into the harness.
//!
//! The fact graph (`qgi2-factgraph`) stores *structure*: typed triples the
//! model proposes and the rules commit. Across turns nothing verbatim
//! survives it, so the agent re-reads files it has already seen. This crate
//! stores the *content*: every sentence of every turn that leaves the window,
//! verbatim, tagged with where it came from and what kind of statement it is,
//! with code, tables and stack traces kept whole as `Block` lines.
//!
//! Three operations, from `QHP-extraction/q-hypergraph/HARNESS-NOTE.md`:
//!
//! - [`LineStore::add`] as a turn leaves the window: split, tag, append.
//!   Deterministic, no model, sub-millisecond per turn.
//! - [`LineStore::retrieve`] on every prompt: term overlap, one-hop expansion
//!   through shared proper-noun terms, per-turn coverage, ±1 neighbours,
//!   rendered as numbered lines under a budget. Deterministic, milliseconds.
//! - Compaction -- the model narrowing the kept set -- is *not* here. It runs
//!   on events only, and only once a counter shows the budget is ever hit.
//!
//! What was measured on the Python prototype (long-chat benchmark, 594
//! cells): full history 0.91, rolling summary 0.85, this memory **0.94** at
//! 123,600 → 7,300 tokens on a large model; 0.76 → **0.88** on a 31B. The
//! deterministic retrieval alone kept every answer-bearing line in a
//! replayed agent run; the model pass only compressed further.
//!
//! Ingestion is the prototype's deterministic ingester ([`parse`]): the
//! sym-tools spaCy service splits and parses each sentence into
//! subject | relation | object with negation and modal, and this crate
//! applies `qhg_extract.py`'s modality map, constituency pattern and role
//! assigner to that parse, persisting the predicate per line and the
//! utterance's negation pairs and discourse links. The lexicon-only path in
//! [`lines`] is the fallback when the service is down, and lines it produced
//! carry the signal `lexicon_fallback`.

pub mod lines;
pub mod parse;
pub mod retrieve;

pub use lines::{Line, LineStore, Role, Speaker};
pub use parse::{ParsedSentence, Predicate};
pub use retrieve::{RetrieveBudget, Retrieved};
