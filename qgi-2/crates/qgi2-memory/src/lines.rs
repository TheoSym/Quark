//! The line store: verbatim, provenance-tagged lines.
//!
//! A line is one sentence of prose, or one whole block of structured content
//! (a fenced code block, a run of table rows, a stack trace). Nothing is
//! paraphrased. The number is a stable identity for retrieval and for
//! re-attaching neighbours; turn and speaker make "the latest statement wins"
//! work without any model reasoning about time; the role lets selection prefer
//! rules and conditions for compliance questions and facts for history.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Who said the line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    User,
    Assistant,
    Tool,
}

impl Speaker {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// What kind of statement a line is. The QLang role vocabulary, assigned by
/// the deterministic rules in [`assign_role`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    Prohibition,
    Obligation,
    Permission,
    TriggerRule,
    Condition,
    Question,
    Cause,
    Requirement,
    Enabler,
    Preventer,
    Attribute,
    Entity,
    Event,
    Sequence,
    Contrast,
    Exception,
    Action,
    Claim,
    /// Code, a table, a listing, JSON, a stack trace: kept whole, never split.
    Block,
}

impl Role {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prohibition => "Prohibition",
            Self::Obligation => "Obligation",
            Self::Permission => "Permission",
            Self::TriggerRule => "TriggerRule",
            Self::Condition => "Condition",
            Self::Question => "Question",
            Self::Cause => "Cause",
            Self::Requirement => "Requirement",
            Self::Enabler => "Enabler",
            Self::Preventer => "Preventer",
            Self::Attribute => "Attribute",
            Self::Entity => "Entity",
            Self::Event => "Event",
            Self::Sequence => "Sequence",
            Self::Contrast => "Contrast",
            Self::Exception => "Exception",
            Self::Action => "Action",
            Self::Claim => "Claim",
            Self::Block => "Block",
        }
    }

    /// Roles a compliance question should prefer.
    pub const fn is_rule(self) -> bool {
        matches!(
            self,
            Self::Prohibition
                | Self::Obligation
                | Self::Permission
                | Self::TriggerRule
                | Self::Condition
        )
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One memory line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Line {
    /// 1-based, stable for the life of the store.
    pub n: u32,
    /// The turn the line came from, 1-based.
    pub turn: u32,
    pub speaker: Speaker,
    pub role: Role,
    /// Verbatim text. For a `Block`, the lines are joined with ` ¦ ` so the
    /// block stays one line in the rendering (the prototype's convention).
    pub text: String,
    /// The parsed predicate, when the sentence went through the service.
    /// `None` for blocks and for the lexicon fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<crate::parse::Predicate>,
    /// The role assigner's confidence and the signal that fired, as the
    /// prototype's `layers.json` records them. `0.0` / empty for blocks.
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub signal: String,
}

impl Line {
    /// The rendering the model sees: `412. [turn 3 user] Fact: ...`. A
    /// multi-line block keeps its newlines below the header line.
    pub fn render(&self) -> String {
        let sep = if self.text.contains('\n') { "\n" } else { " " };
        format!(
            "{}. [turn {} {}] {}:{sep}{}",
            self.n,
            self.turn,
            self.speaker.as_str(),
            self.role,
            self.text
        )
    }
}

/// The store. Append-only; retrieval never rewrites it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LineStore {
    pub lines: Vec<Line>,
    /// Lines a compaction event kept. Persisted; later events see only new
    /// candidates.
    #[serde(default)]
    pub kept: std::collections::BTreeSet<u32>,
    /// Lines a compaction event has judged, kept or not. A judged line that
    /// was not kept is excluded from retrieval; the store itself is never
    /// rewritten, so a later question can re-retrieve from the full store by
    /// clearing this set.
    #[serde(default)]
    pub judged: std::collections::BTreeSet<u32>,
    /// `(affirming line, negating line)` pairs the ingester derived: same
    /// subject, opposite negation, two or more shared content words.
    #[serde(default)]
    pub negation_pairs: Vec<(u32, u32)>,
    /// `(from, to, relation)` discourse links from a sentence's opening
    /// connective to its predecessor.
    #[serde(default)]
    pub discourse_links: Vec<(u32, u32, String)>,
}

impl LineStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn get(&self, n: u32) -> Option<&Line> {
        // n is 1-based and dense, so the index is n-1; the check guards a
        // store loaded from a file that was hand-edited.
        self.lines
            .get((n as usize).wrapping_sub(1))
            .filter(|l| l.n == n)
            .or_else(|| self.lines.iter().find(|l| l.n == n))
    }

    /// Add one utterance of a turn. Prose is split into sentences and role
    /// tagged; blocks are kept whole. Returns the ids of the new lines.
    ///
    /// `max_chars` caps one utterance (the note's default for a tool
    /// observation is 12,000 characters before it enters the store); the
    /// head is kept, since a tool result's structure is at its head.
    pub fn add(&mut self, turn: u32, speaker: Speaker, text: &str, max_chars: usize) -> Vec<u32> {
        let text: String = text.chars().take(max_chars).collect();
        let mut ids = Vec::new();
        for (segment, is_block) in split_blocks(&text) {
            if is_block {
                let joined = segment
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ¦ ");
                ids.push(self.push(turn, speaker, Role::Block, joined));
                continue;
            }
            for sentence in split_sentences(&segment) {
                let role = assign_role(&sentence);
                ids.push(self.push(turn, speaker, role, sentence));
            }
        }
        ids
    }

    /// Add one utterance as a single verbatim `Block` line, newlines kept.
    ///
    /// For tool output. A file read, a listing, a test run: splitting those
    /// into sentences and collapsing whitespace destroys exactly what a later
    /// `edit` needs to match. On the A/B bench the planner re-read the same
    /// file five rounds running because the memory's copy no longer matched
    /// the file byte for byte.
    pub fn add_block(&mut self, turn: u32, speaker: Speaker, text: &str, max_chars: usize) -> u32 {
        let text: String = text.chars().take(max_chars).collect();
        self.push(turn, speaker, Role::Block, text.trim_end().to_string())
    }

    fn push(&mut self, turn: u32, speaker: Speaker, role: Role, text: String) -> u32 {
        let n = self.lines.len() as u32 + 1;
        self.lines.push(Line {
            n,
            turn,
            speaker,
            role,
            text,
            predicate: None,
            confidence: 0.0,
            signal: if role == Role::Block { String::new() } else { "lexicon_fallback".into() },
        });
        n
    }

    /// Record a compaction event: every line in `judged` has now been seen by
    /// the compactor, and those in `kept` survive. Lines already judged are
    /// never re-judged.
    pub fn record_compaction(&mut self, judged: &[u32], kept: &[u32]) {
        self.judged.extend(judged.iter().copied());
        self.kept.extend(kept.iter().copied());
    }

    /// Whether retrieval may render this line: unjudged, or judged and kept.
    pub fn admits(&self, n: u32) -> bool {
        !self.judged.contains(&n) || self.kept.contains(&n)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Split a turn into `(segment, is_block)` runs: fenced code, runs of table
/// rows (`|`-led lines), and traceback-shaped runs are blocks; everything
/// else is prose. A port of the prototype's `split_blocks`, plus the
/// traceback case, which the note lists as a Block but the prototype left to
/// the sentence splitter.
pub fn split_blocks(text: &str) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();
    let mut prose: Vec<&str> = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let mut fence = false;
    let mut traceback = false;

    let flush_prose = |prose: &mut Vec<&str>, out: &mut Vec<(String, bool)>| {
        if !prose.is_empty() {
            out.push((prose.join("\n"), false));
            prose.clear();
        }
    };
    let flush_block = |block: &mut Vec<&str>, out: &mut Vec<(String, bool)>| {
        if !block.is_empty() {
            out.push((block.join("\n"), true));
            block.clear();
        }
    };

    for line in text.lines() {
        let s = line.trim();
        if s.starts_with("```") {
            if fence {
                block.push(line);
                flush_block(&mut block, &mut out);
                fence = false;
            } else {
                flush_prose(&mut prose, &mut out);
                fence = true;
                block.push(line);
            }
            continue;
        }
        if fence {
            block.push(line);
            continue;
        }
        if s.starts_with("Traceback (most recent call last)") {
            flush_prose(&mut prose, &mut out);
            traceback = true;
            block.push(line);
            continue;
        }
        if traceback {
            // A traceback ends at the first line that is neither indented nor
            // an error line, after the error line itself.
            block.push(line);
            let is_error_line = !line.starts_with(' ') && s.contains(':');
            if is_error_line {
                flush_block(&mut block, &mut out);
                traceback = false;
            }
            continue;
        }
        if s.starts_with('|') {
            flush_prose(&mut prose, &mut out);
            block.push(line);
            continue;
        }
        flush_block(&mut block, &mut out);
        prose.push(line);
    }
    flush_block(&mut block, &mut out);
    flush_prose(&mut prose, &mut out);
    out.into_iter()
        .filter(|(seg, _)| !seg.trim().is_empty())
        .collect()
}

/// Split prose into sentences. Deterministic and conservative: a sentence
/// ends at `.`, `!` or `?` followed by whitespace and an uppercase letter,
/// a digit, a backtick or a quote, or at a blank line. Abbreviations like
/// `e.g.` and version numbers (`v0.81.3`) do not split because what follows
/// them is lowercase or a digit adjacent to the period.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n") {
        let flat: String = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.is_empty() {
            continue;
        }
        let chars: Vec<char> = flat.chars().collect();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if matches!(c, '.' | '!' | '?') {
                let next_ws = chars.get(i + 1).is_some_and(|c| c.is_whitespace());
                let after = chars.get(i + 2).copied();
                let starts_new = after.is_some_and(|a| {
                    a.is_uppercase() || a.is_ascii_digit() || a == '`' || a == '"' || a == '\''
                });
                let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
                if next_ws
                    && starts_new
                    && !(c == '.' && prev_is_digit && after.is_some_and(|a| a.is_ascii_digit()))
                {
                    let s: String = chars[start..=i].iter().collect();
                    if !s.trim().is_empty() {
                        out.push(s.trim().to_string());
                    }
                    start = i + 2;
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }
        if start < chars.len() {
            let s: String = chars[start..].iter().collect();
            if !s.trim().is_empty() {
                out.push(s.trim().to_string());
            }
        }
    }
    out
}

// ---- sym-ingest role assigner, ported from qhg_extract.py -------------------

const CAUSAL: &[&str] = &[
    "causes",
    "leads",
    "results",
    "produces",
    "generates",
    "triggers",
];
const REQUIREMENT: &[&str] = &["requires", "depends", "needs", "demands", "necessitates"];
const ENABLER: &[&str] = &["enables", "supports", "allows", "facilitates", "empowers"];
const PREVENTER: &[&str] = &["prevents", "blocks", "inhibits", "stops", "prohibits"];
const ATTRIBUTE: &[&str] = &[
    "has",
    "have",
    "contains",
    "includes",
    "stores",
    "holds",
    "comprises",
];
const IDENTITY: &[&str] = &["is", "are", "represents", "constitutes", "means", "denotes"];
const EVENT: &[&str] = &[
    "occurs", "happens", "arrives", "appears", "emerges", "returns",
];
const TEMPORAL: &[&str] = &[
    "first",
    "then",
    "next",
    "finally",
    "after",
    "before",
    "subsequently",
    "starts",
    "begins",
    "ends",
    "follows",
];
const CONTRAST: &[&str] = &[
    "however",
    "but",
    "although",
    "nevertheless",
    "in contrast",
    "on the other hand",
    "conversely",
    "yet",
];
const MODALS: &[&str] = &["must", "shall", "should", "may", "can", "could", "would"];
const NEGATORS: &[&str] = &[
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "n't",
    "without",
    "neither",
    "nor",
    "forbidden",
];

fn modality_of(modal: Option<&str>, negated: bool) -> &'static str {
    let Some(m) = modal else { return "fact" };
    if negated && matches!(m, "can" | "must" | "shall") {
        return "prohibition";
    }
    match m {
        "must" | "shall" => "obligation",
        "should" | "would" => "recommendation",
        "may" | "can" | "could" => "permission",
        _ => "fact",
    }
}

fn constituency_of(text: &str) -> &'static str {
    let t = text.trim();
    if t.ends_with('?') {
        return "question";
    }
    let lower = t.to_lowercase();
    let starts = ["if ", "when ", "unless ", "whenever ", "provided that "];
    if starts.iter().any(|s| lower.starts_with(s)) {
        return "conditional";
    }
    for mid in [" if ", " unless ", " provided that "] {
        if lower.contains(mid) {
            return "conditional";
        }
    }
    "simple"
}

/// The first verb-like token, by lexicon, standing in for the parser's root
/// verb; and the first modal; and whether a negator is present.
fn scan(text: &str) -> (Option<&'static str>, Option<&'static str>, bool) {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|t| !t.is_empty())
        .collect();
    let mut modal = None;
    let mut verb = None;
    let mut negated = false;
    for t in &tokens {
        if modal.is_none()
            && let Some(m) = MODALS.iter().find(|m| *m == t)
        {
            modal = Some(*m);
        }
        if NEGATORS.iter().any(|n| n == t) || t.ends_with("n't") {
            negated = true;
        }
        if verb.is_none() {
            for lex in [
                CAUSAL,
                REQUIREMENT,
                ENABLER,
                PREVENTER,
                ATTRIBUTE,
                IDENTITY,
                EVENT,
            ] {
                if let Some(v) = lex.iter().find(|v| *v == t) {
                    verb = Some(*v);
                    break;
                }
            }
        }
    }
    (verb, modal, negated)
}

/// sym-ingest's role assigner, in its original priority order.
pub fn assign_role(text: &str) -> Role {
    let (verb, modal, negated) = scan(text);
    let modality = modality_of(modal, negated);
    let constituency = constituency_of(text);
    let lower = text.to_lowercase();

    match modality {
        "prohibition" => return Role::Prohibition,
        "obligation" => return Role::Obligation,
        "permission" => return Role::Permission,
        _ => {}
    }
    if constituency == "conditional" {
        return if modality != "fact" {
            Role::TriggerRule
        } else {
            Role::Condition
        };
    }
    if constituency == "question" {
        return Role::Question;
    }
    if let Some(v) = verb {
        if CAUSAL.contains(&v) {
            return Role::Cause;
        }
        if REQUIREMENT.contains(&v) {
            return Role::Requirement;
        }
        if ENABLER.contains(&v) {
            return Role::Enabler;
        }
        if PREVENTER.contains(&v) {
            return Role::Preventer;
        }
        if ATTRIBUTE.contains(&v) {
            return Role::Attribute;
        }
        if IDENTITY.contains(&v) {
            return Role::Entity;
        }
        if EVENT.contains(&v) {
            return Role::Event;
        }
    }
    if TEMPORAL
        .iter()
        .any(|s| lower.split(|c: char| !c.is_alphanumeric()).any(|t| t == *s))
    {
        return Role::Sequence;
    }
    if CONTRAST.iter().any(|s| lower.starts_with(s)) {
        return Role::Contrast;
    }
    if lower.contains("unless") || lower.contains("except") {
        return Role::Exception;
    }
    // sym: "active verb with agent" when a subject and relation were parsed;
    // without a parser, a sentence with a recognisable verb-ish word after
    // its first token is treated as an action, else a claim.
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() >= 3
        && words
            .iter()
            .skip(1)
            .any(|w| w.ends_with('s') || w.ends_with("ed"))
    {
        return Role::Action;
    }
    Role::Claim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_becomes_role_tagged_sentences_and_blocks_stay_whole() {
        let mut s = LineStore::new();
        let ids = s.add(
            1,
            Speaker::Tool,
            "The borrower must repay the loan within 30 days. A missed payment triggers the \
             delinquency process.\n\n```rust\nfn a() {}\nfn b() {}\n```\n\n| item | qty |\n| rice | 1 |",
            12_000,
        );
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(s.get(1).unwrap().role, Role::Obligation);
        assert_eq!(s.get(2).unwrap().role, Role::Cause);
        assert_eq!(s.get(3).unwrap().role, Role::Block);
        assert!(s.get(3).unwrap().text.contains("fn a() {} ¦ fn b() {}"));
        assert_eq!(s.get(4).unwrap().role, Role::Block);
        assert!(s.get(4).unwrap().text.starts_with("| item"));
    }

    #[test]
    fn the_role_assigner_matches_the_prototypes_priority() {
        assert_eq!(
            assign_role("You must not delete the audit log."),
            Role::Prohibition
        );
        assert_eq!(
            assign_role("The service must restart nightly."),
            Role::Obligation
        );
        assert_eq!(
            assign_role("Operators may override the schedule."),
            Role::Permission
        );
        assert_eq!(
            assign_role("If the score falls below 600, manual review applies."),
            Role::Condition
        );
        // Modality wins over constituency, as in the prototype: a `must` in a
        // conditional is an Obligation; a recommendation in one is a trigger.
        assert_eq!(assign_role("If it fails you must retry."), Role::Obligation);
        assert_eq!(
            assign_role("If it fails you should retry."),
            Role::TriggerRule
        );
        assert_eq!(
            assign_role("Does the handler check expiry?"),
            Role::Question
        );
        assert_eq!(
            assign_role("The module depends on jsonwebtoken."),
            Role::Requirement
        );
        assert_eq!(
            assign_role("The struct contains three fields."),
            Role::Attribute
        );
        assert_eq!(assign_role("Auth is a module."), Role::Entity);
        assert_eq!(
            assign_role("First we parse, then we validate."),
            Role::Sequence
        );
        assert_eq!(assign_role("However, the cache was cold."), Role::Contrast);
    }

    #[test]
    fn tracebacks_are_one_block() {
        let text = "Ran the tests.\nTraceback (most recent call last):\n  File \"ingest.py\", line 88\nKeyError: 'sku'\nThen it stopped.";
        let parts = split_blocks(text);
        assert_eq!(parts.len(), 3, "{parts:?}");
        assert!(parts[1].1, "traceback is a block");
        assert!(parts[1].0.contains("KeyError"));
        assert!(!parts[2].1);
    }

    #[test]
    fn sentence_splitting_does_not_break_versions_or_abbreviations() {
        let s = split_sentences("Upgrade to v0.81.3 first. Then run e.g. the smoke test. Done?");
        assert_eq!(
            s,
            vec![
                "Upgrade to v0.81.3 first.".to_string(),
                "Then run e.g. the smoke test.".to_string(),
                "Done?".to_string()
            ]
        );
    }

    #[test]
    fn a_block_keeps_its_bytes_and_newlines() {
        let mut s = LineStore::new();
        let src = "     1\tdef slugify(s):\n     2\t    return s.lower()   \n";
        let n = s.add_block(2, Speaker::Tool, src, 12_000);
        let l = s.get(n).unwrap();
        assert_eq!(l.role, Role::Block);
        assert_eq!(l.text, src.trim_end());
        let r = l.render();
        assert!(
            r.starts_with("1. [turn 2 tool] Block:\n     1\tdef slugify"),
            "{r}"
        );
    }

    #[test]
    fn rendering_carries_provenance() {
        let mut s = LineStore::new();
        s.add(3, Speaker::User, "I switched to Costco last month.", 100);
        assert!(s.get(1).unwrap().render().starts_with("1. [turn 3 user] "));
    }

    #[test]
    fn the_store_round_trips_through_json() {
        let mut s = LineStore::new();
        s.add(1, Speaker::Assistant, "Alpha beta. Gamma delta!", 100);
        let back = LineStore::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(back, s);
    }
}
