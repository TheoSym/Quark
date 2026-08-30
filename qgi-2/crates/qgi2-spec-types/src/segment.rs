//! Prompt segments: the fixed order that makes the prompt cache-shaped.
//!
//! Spec invariant:
//!
//! > Prompt order is always `core → mood → durable slice → active skills →
//! > session subgraph → query`. Segments 1–3 are byte-stable within a session;
//! > 4–6 are the only recomputed tokens.
//!
//! The order is a property of the type, not of each call site: [`SEGMENT_ORDER`]
//! is the single source of truth and [`SegmentSet`] can only be rendered
//! through it. That is what keeps "stable bytes at the front, volatile bytes at
//! the end" true by construction instead of by convention.

use serde::{Deserialize, Serialize};
use std::fmt;

/// One of the six prompt segments, in prompt order.
///
/// The discriminants are the spec's 1-based positions so that
/// `SegmentId::Core < SegmentId::Query` orders segments correctly and a
/// serialized ordinal survives a round trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SegmentId {
    /// 1. The invariant harness prompt. Identical for every session.
    Core = 1,
    /// 2. Mood preamble: traversal, conflict policy, and tool framing.
    Mood = 2,
    /// 3. Durable memory slice promoted from previous sessions.
    Durable = 3,
    /// 4. Skills activated for this turn.
    Skills = 4,
    /// 5. The session subgraph retrieved for this turn.
    Subgraph = 5,
    /// 6. The user's query.
    Query = 6,
}

/// Every segment, in the exact order the spec mandates.
///
/// Assembly walks this constant; no other ordering exists in the codebase.
pub const SEGMENT_ORDER: [SegmentId; 6] = [
    SegmentId::Core,
    SegmentId::Mood,
    SegmentId::Durable,
    SegmentId::Skills,
    SegmentId::Subgraph,
    SegmentId::Query,
];

impl SegmentId {
    /// Whether this segment is byte-stable for the lifetime of a session.
    ///
    /// Stable segments form the cached prefix. A change to one invalidates
    /// every block after it, which is why the assembler reports a stable-prefix
    /// change as a cache incident rather than a routine recompute.
    pub const fn is_stable(self) -> bool {
        matches!(self, Self::Core | Self::Mood | Self::Durable)
    }

    /// Whether this segment is expected to be recomputed each turn.
    pub const fn is_volatile(self) -> bool {
        !self.is_stable()
    }

    /// 1-based position in the prompt.
    pub const fn position(self) -> u8 {
        self as u8
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Mood => "mood",
            Self::Durable => "durable",
            Self::Skills => "skills",
            Self::Subgraph => "subgraph",
            Self::Query => "query",
        }
    }
}

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A BLAKE3 hash of one segment's rendered bytes.
///
/// Hashes are what let the harness answer "did the cached prefix survive?"
/// without diffing prompt text, and they are cheap enough to compute every
/// turn for all six segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentHash(pub [u8; 32]);

impl SegmentHash {
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    /// Short form for logs and the UI, where a full 64 hex chars is noise.
    pub fn short(self) -> String {
        self.to_hex()[..12].to_string()
    }
}

impl fmt::Display for SegmentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.short())
    }
}

/// One rendered segment: its identity, its bytes, and the hash of those bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub id: SegmentId,
    pub text: String,
    pub hash: SegmentHash,
}

impl Segment {
    /// Build a segment, hashing its text.
    ///
    /// The hash covers the segment id as well as the text so that two segments
    /// which happen to render identical bytes (an empty `skills` and an empty
    /// `subgraph`, say) do not collide in the cache-hit bookkeeping.
    pub fn new(id: SegmentId, text: impl Into<String>) -> Self {
        let text = text.into();
        let mut hasher = blake3_hasher();
        hasher.update(&[id.position()]);
        hasher.update(text.as_bytes());
        Self {
            id,
            hash: SegmentHash(*hasher.finalize().as_bytes()),
            text,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

// Behind one function so every segment hash is constructed identically; the
// domain separation in `Segment::new` is only sound if nothing else builds a
// hasher its own way.
fn blake3_hasher() -> blake3::Hasher {
    blake3::Hasher::new()
}

/// The six segments of one assembled prompt.
///
/// Constructed only through [`SegmentSet::new`], which takes the segments in
/// spec order, so an out-of-order prompt cannot be represented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentSet {
    segments: [Segment; 6],
}

impl SegmentSet {
    /// Assemble the six segments in spec order.
    pub fn new(
        core: String,
        mood: String,
        durable: String,
        skills: String,
        subgraph: String,
        query: String,
    ) -> Self {
        Self {
            segments: [
                Segment::new(SegmentId::Core, core),
                Segment::new(SegmentId::Mood, mood),
                Segment::new(SegmentId::Durable, durable),
                Segment::new(SegmentId::Skills, skills),
                Segment::new(SegmentId::Subgraph, subgraph),
                Segment::new(SegmentId::Query, query),
            ],
        }
    }

    pub fn get(&self, id: SegmentId) -> &Segment {
        // SEGMENT_ORDER is dense and 1-based, so the ordinal indexes directly.
        &self.segments[(id.position() - 1) as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter()
    }

    /// The stable prefix: segments 1–3, which must not change within a session.
    pub fn stable_prefix(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(|s| s.id.is_stable())
    }

    /// The volatile tail: segments 4–6, the only tokens expected to recompute.
    pub fn volatile_tail(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(|s| s.id.is_volatile())
    }

    /// Concatenated prompt text in spec order.
    ///
    /// Segments are joined with a single newline and empty segments still emit
    /// their separator: dropping an empty segment would shift every later byte
    /// and silently invalidate the cached prefix the first time a session had
    /// no active skills.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.byte_len() + SEGMENT_ORDER.len());
        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&seg.text);
        }
        out
    }

    /// Just the stable prefix, rendered. This is what the engine should see as
    /// the cacheable system prompt.
    pub fn render_stable(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.stable_prefix().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&seg.text);
        }
        out
    }

    /// Just the volatile tail, rendered.
    pub fn render_volatile(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.volatile_tail().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&seg.text);
        }
        out
    }

    pub fn byte_len(&self) -> usize {
        self.segments.iter().map(|s| s.text.len()).sum()
    }

    /// All six hashes, in order, for logging and cache diffing.
    pub fn hashes(&self) -> [(SegmentId, SegmentHash); 6] {
        std::array::from_fn(|i| (self.segments[i].id, self.segments[i].hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SegmentSet {
        SegmentSet::new(
            "CORE".into(),
            "MOOD".into(),
            "DURABLE".into(),
            "SKILLS".into(),
            "SUBGRAPH".into(),
            "QUERY".into(),
        )
    }

    #[test]
    fn segments_render_in_spec_order() {
        assert_eq!(
            sample().render(),
            "CORE\nMOOD\nDURABLE\nSKILLS\nSUBGRAPH\nQUERY"
        );
    }

    #[test]
    fn stable_prefix_is_segments_one_through_three() {
        let s = sample();
        let ids: Vec<_> = s.stable_prefix().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![SegmentId::Core, SegmentId::Mood, SegmentId::Durable]
        );
        assert_eq!(s.render_stable(), "CORE\nMOOD\nDURABLE");
    }

    #[test]
    fn volatile_tail_is_segments_four_through_six() {
        let s = sample();
        let ids: Vec<_> = s.volatile_tail().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![SegmentId::Skills, SegmentId::Subgraph, SegmentId::Query]
        );
    }

    #[test]
    fn same_bytes_hash_the_same() {
        assert_eq!(sample().hashes(), sample().hashes());
    }

    #[test]
    fn changing_the_tail_leaves_the_stable_prefix_hashes_untouched() {
        let a = sample();
        let b = SegmentSet::new(
            "CORE".into(),
            "MOOD".into(),
            "DURABLE".into(),
            "SKILLS".into(),
            "SUBGRAPH".into(),
            "A DIFFERENT QUERY".into(),
        );
        for id in [SegmentId::Core, SegmentId::Mood, SegmentId::Durable] {
            assert_eq!(a.get(id).hash, b.get(id).hash, "{id} should be stable");
        }
        assert_ne!(a.get(SegmentId::Query).hash, b.get(SegmentId::Query).hash);
    }

    #[test]
    fn identical_text_in_different_segments_does_not_collide() {
        // An empty skills segment and an empty subgraph segment must be
        // distinguishable, or cache bookkeeping conflates them.
        let s = SegmentSet::new(
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        );
        assert_ne!(s.get(SegmentId::Skills).hash, s.get(SegmentId::Subgraph).hash);
    }

    #[test]
    fn empty_segments_still_emit_separators() {
        // Otherwise the first turn with no active skills shifts every later
        // byte and invalidates the cached prefix.
        let s = SegmentSet::new(
            "CORE".into(),
            "MOOD".into(),
            "DURABLE".into(),
            String::new(),
            "SUBGRAPH".into(),
            "QUERY".into(),
        );
        assert_eq!(s.render(), "CORE\nMOOD\nDURABLE\n\nSUBGRAPH\nQUERY");
    }
}
