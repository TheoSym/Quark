//! The deterministic ingester, as in `QHP-extraction/q-hypergraph/qhg_extract.py`.
//!
//! The parse itself -- subject, relation, object, negation, modal -- comes
//! from the sym-tools spaCy service (`QHP-CORE-1/services/sym-tools`,
//! `/sentences` and `/batch-parse`). Everything downstream of the parse is
//! here, ported line for line: the modality map, the constituency pattern,
//! sym's role assigner in its original priority order, and the
//! document-level derivations the prototype persists rather than discards:
//! negation pairs and discourse links.
//!
//! The lexicon-only path in [`crate::lines::assign_role`] remains as the
//! fallback for when the service is unreachable, and is marked as such on
//! the lines it produces.

use crate::lines::{Line, LineStore, Role, Speaker};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One sentence's predicate, the sym shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Predicate {
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub relation: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub negated: bool,
    #[serde(default)]
    pub modal: Option<String>,
}

/// A parsed sentence ready for the store.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSentence {
    pub text: String,
    pub predicate: Predicate,
}

// ---- sym-ingest semantics, ported verbatim from qhg_extract.py ----------

const CAUSAL: &[&str] = &["causes", "leads", "results", "produces", "generates", "triggers"];
const REQUIREMENT: &[&str] = &["requires", "depends", "needs", "demands", "necessitates"];
const ENABLER: &[&str] = &["enables", "supports", "allows", "facilitates", "empowers"];
const PREVENTER: &[&str] = &["prevents", "blocks", "inhibits", "stops", "prohibits"];
const ATTRIBUTE: &[&str] = &["has", "have", "contains", "includes", "stores", "holds", "comprises"];
const IDENTITY: &[&str] = &["is", "are", "represents", "constitutes", "means", "denotes"];
const EVENT: &[&str] = &["occurs", "happens", "arrives", "appears", "emerges", "returns"];
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

/// `MODALITY_MAP` plus the prohibition rule.
pub fn modality_of(modal: Option<&str>, negated: bool) -> &'static str {
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

/// `constituency_of`: question, conditional (`COND_RE`), or simple.
pub fn constituency_of(text: &str) -> &'static str {
    let t = text.trim();
    if t.ends_with('?') {
        return "question";
    }
    let lower = t.to_lowercase();
    for s in ["if ", "when ", "unless ", "whenever ", "provided that "] {
        if lower.starts_with(s) {
            return "conditional";
        }
    }
    for mid in [" if ", " unless ", " provided that "] {
        if lower.contains(mid) {
            return "conditional";
        }
    }
    "simple"
}

/// `NEG_RE`: whether a negator appears anywhere in the sentence.
pub fn negator_present(text: &str) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .collect();
    let single = [
        "not", "no", "never", "cannot", "can't", "forbidden", "without", "neither", "nor",
    ];
    if words.iter().any(|w| single.contains(w) || w.starts_with("prohibit")) {
        return true;
    }
    lower.contains("must not") || lower.contains("may not")
}

/// sym's role assigner over a parsed predicate: `assign_role(pred, constituency)`
/// with the same priority order and the same `Action` condition (a parsed
/// subject and relation, and the verb not an identity verb).
pub fn assign_role(pred: &Predicate, constituency: &str) -> (Role, f32, &'static str) {
    let verb = pred.relation.to_lowercase();
    let verb = verb.trim();
    let first = verb.split_whitespace().next().unwrap_or("");
    let full = format!("{} {} {}", pred.subject, pred.relation, pred.object).to_lowercase();
    let m = modality_of(pred.modal.as_deref(), pred.negated);

    if m == "prohibition" {
        return (Role::Prohibition, 0.95, "modal+negation=prohibition");
    }
    if m == "obligation" {
        return (Role::Obligation, 0.95, "modal=obligation");
    }
    if m == "permission" {
        return (Role::Permission, 0.90, "modal=permission");
    }
    if constituency == "conditional" {
        return if m != "fact" {
            (Role::TriggerRule, 0.85, "SBAR+modal")
        } else {
            (Role::Condition, 0.85, "SBAR(if/when)")
        };
    }
    if constituency == "question" {
        return (Role::Question, 0.95, "question_mark");
    }
    if CAUSAL.contains(&first) {
        return (Role::Cause, 0.85, "causal_verb");
    }
    if REQUIREMENT.contains(&first) {
        return (Role::Requirement, 0.85, "requirement_verb");
    }
    if ENABLER.contains(&first) {
        return (Role::Enabler, 0.85, "enabler_verb");
    }
    if PREVENTER.contains(&first) {
        return (Role::Preventer, 0.85, "preventer_verb");
    }
    if ATTRIBUTE.contains(&first) {
        return (Role::Attribute, 0.90, "attribute_verb");
    }
    if IDENTITY.contains(&first) {
        return (Role::Entity, 0.80, "identity_verb");
    }
    if EVENT.contains(&first) {
        return (Role::Event, 0.80, "event_verb");
    }
    for s in TEMPORAL {
        if full.contains(s) {
            return (Role::Sequence, 0.80, "temporal_signal");
        }
    }
    for s in CONTRAST {
        if full.starts_with(s) {
            return (Role::Contrast, 0.80, "contrast_connective");
        }
    }
    if full.contains("unless") || full.contains("except") {
        return (Role::Exception, 0.75, "exception_keyword");
    }
    if !pred.subject.is_empty() && !pred.relation.is_empty() && !IDENTITY.contains(&first) {
        return (Role::Action, 0.60, "active_verb_with_agent");
    }
    (Role::Claim, 0.30, "default")
}

/// `DISC`: the discourse relation a sentence opens with, if any.
pub fn discourse_of(text: &str) -> Option<&'static str> {
    let t = text.trim().to_lowercase();
    let table: &[(&str, &[&str])] = &[
        (
            "cause",
            &["because", "since", "therefore", "thus", "hence", "so that", "as a result", "consequently"],
        ),
        (
            "contrast",
            &["however", "but", "although", "nevertheless", "yet", "in contrast", "on the other hand"],
        ),
        ("example", &["for example", "for instance", "e.g."]),
        (
            "elaboration",
            &["also", "moreover", "furthermore", "in addition", "that is", "i.e."],
        ),
        ("summary", &["in summary", "overall", "in conclusion", "to summarize"]),
    ];
    for (rel, starts) in table {
        for s in *starts {
            if t.starts_with(s) && t[s.len()..].chars().next().is_none_or(|c| !c.is_alphanumeric()) {
                return Some(rel);
            }
        }
    }
    None
}

fn content_words4(text: &str) -> std::collections::BTreeSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|w| w.len() >= 4)
        .map(str::to_string)
        .collect()
}

impl LineStore {
    /// Add one utterance already split and parsed by the service. Returns the
    /// new line ids. Persists what the prototype persists: the predicate on
    /// each line, and the utterance's negation pairs and discourse links.
    pub fn add_parsed(&mut self, turn: u32, speaker: Speaker, sentences: &[ParsedSentence]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(sentences.len());
        for s in sentences {
            let cons = constituency_of(&s.text);
            let (role, confidence, signal) = assign_role(&s.predicate, cons);
            let n = self.lines.len() as u32 + 1;
            self.lines.push(Line {
                n,
                turn,
                speaker,
                role,
                text: s.text.clone(),
                predicate: Some(s.predicate.clone()),
                confidence,
                signal: signal.to_string(),
            });
            ids.push(n);
        }

        // Document-level derivations, over this utterance only, as the
        // prototype does per document: negation pairs between sentences that
        // share a subject and disagree on negation with >= 2 shared content
        // words; discourse links from each sentence's opening connective to
        // its predecessor.
        let mut by_subject: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for n in &ids {
            let l = &self.lines[(*n - 1) as usize];
            if let Some(p) = &l.predicate {
                let sj = p.subject.to_lowercase();
                if !sj.is_empty() {
                    by_subject.entry(sj).or_default().push(*n);
                }
            }
        }
        for group in by_subject.values() {
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let a = &self.lines[(group[i] - 1) as usize];
                    let b = &self.lines[(group[j] - 1) as usize];
                    let pa = a.predicate.as_ref().is_some_and(|p| p.negated) || negator_present(&a.text);
                    let pb = b.predicate.as_ref().is_some_and(|p| p.negated) || negator_present(&b.text);
                    if pa != pb {
                        let shared = content_words4(&a.text)
                            .intersection(&content_words4(&b.text))
                            .count();
                        if shared >= 2 {
                            let (affirm, negate) = if pb { (a.n, b.n) } else { (b.n, a.n) };
                            self.negation_pairs.push((affirm, negate));
                        }
                    }
                }
            }
        }
        for w in ids.windows(2) {
            let cur = &self.lines[(w[1] - 1) as usize];
            if let Some(rel) = discourse_of(&cur.text) {
                self.discourse_links.push((w[0], w[1], rel.to_string()));
            }
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(s: &str, r: &str, o: &str, negated: bool, modal: Option<&str>) -> Predicate {
        Predicate {
            subject: s.into(),
            relation: r.into(),
            object: o.into(),
            negated,
            modal: modal.map(str::to_string),
        }
    }

    #[test]
    fn the_role_assigner_follows_syms_priority_over_a_real_parse() {
        // The service's own output for these two sentences, recorded.
        let p1 = pred("The borrower", "repay", "the loan", false, Some("must"));
        assert_eq!(assign_role(&p1, "simple").0, Role::Obligation);
        let p2 = pred("A missed payment", "triggers", "the delinquency process", false, None);
        assert_eq!(assign_role(&p2, "simple").0, Role::Cause);
        // Prohibition: modal + negation.
        assert_eq!(assign_role(&pred("You", "delete", "logs", true, Some("must")), "simple").0, Role::Prohibition);
        // Conditional with a recommendation is a trigger rule; plain is a condition.
        assert_eq!(assign_role(&pred("it", "retry", "", false, Some("should")), "conditional").0, Role::TriggerRule);
        assert_eq!(assign_role(&pred("score", "falls", "", false, None), "conditional").0, Role::Condition);
        // Identity verb is an Entity, not an Action.
        assert_eq!(assign_role(&pred("Auth", "is", "a module", false, None), "simple").0, Role::Entity);
        // Parsed subject + non-identity verb, no lexicon hit: Action; nothing parsed: Claim.
        assert_eq!(assign_role(&pred("We", "refactored", "the loop", false, None), "simple").0, Role::Action);
        assert_eq!(assign_role(&pred("", "", "", false, None), "simple").0, Role::Claim);
    }

    #[test]
    fn negation_pairs_and_discourse_links_are_persisted() {
        let mut s = LineStore::new();
        let ids = s.add_parsed(
            1,
            Speaker::User,
            &[
                ParsedSentence {
                    text: "The cache serves cached pages quickly.".into(),
                    predicate: pred("The cache", "serves", "cached pages", false, None),
                },
                ParsedSentence {
                    text: "However, the cache does not serve cached pages on a cold start.".into(),
                    predicate: pred("the cache", "serve", "cached pages", true, None),
                },
            ],
        );
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(s.negation_pairs, vec![(1, 2)], "affirm then negate");
        assert_eq!(s.discourse_links, vec![(1, 2, "contrast".to_string())]);
        assert!(s.get(2).unwrap().predicate.as_ref().unwrap().negated);
    }

    #[test]
    fn constituency_and_modality_match_the_prototype() {
        assert_eq!(constituency_of("If it fails, retry."), "conditional");
        assert_eq!(constituency_of("Retry unless it passes."), "conditional");
        assert_eq!(constituency_of("Does it pass?"), "question");
        assert_eq!(constituency_of("It passes."), "simple");
        assert_eq!(modality_of(Some("can"), true), "prohibition");
        assert_eq!(modality_of(Some("should"), true), "recommendation");
        assert_eq!(modality_of(None, true), "fact");
        assert!(negator_present("You must not do that"));
        assert!(negator_present("It prohibits access"));
        assert!(!negator_present("It allows access"));
        assert_eq!(discourse_of("For example, a table."), Some("example"));
        assert_eq!(discourse_of("Format the disk."), None);
    }
}
