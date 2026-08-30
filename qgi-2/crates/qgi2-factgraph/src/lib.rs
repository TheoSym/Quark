//! The typed fact graph.
//!
//! Spec:
//!
//! > Memory — jcode's in-RAM graph, extended with typed facts.
//! > Keep memory small and typed. A fact graph in RAM, rendered
//! > deterministically, so the volatile tail stays short.
//!
//! and the invariant that pins the whole cache strategy:
//!
//! > Rendering is deterministic: same graph → same bytes → same cache blocks.
//!
//! Determinism here is not "we sort before rendering" as an afterthought. The
//! graph stores facts in a [`BTreeMap`] keyed by [`FactId`], every index is a
//! [`BTreeMap`]/[`BTreeSet`], and traversal visits neighbours in key order. A
//! `HashMap` anywhere on the render path would make the prompt depend on hash
//! seed, which changes per process and would silently destroy the prefix cache
//! across restarts.

pub mod render;
pub mod retrieval;
pub mod store;
pub mod traversal;

pub use render::{RenderBudget, RenderedSubgraph};
pub use retrieval::{EntryPoint, Retrieval};
pub use store::{CommitOutcome, FactGraph, Scope};
pub use traversal::{TraversalResult, Walk};
