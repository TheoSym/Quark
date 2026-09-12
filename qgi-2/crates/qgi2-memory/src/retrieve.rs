//! Deterministic retrieval over the line store.
//!
//! The recipe that won Phase F/G on the prototype (`select_for_question.py`,
//! `run_phase_d.datalog_select`), without the model pass:
//!
//! 1. **Term overlap.** Score every line by shared content words with the
//!    query (stop words out, light stemming). Seed = the top `k_pre` hits.
//! 2. **One-hop expansion.** Proper-noun-shaped terms (`Costco`, `auth.rs`,
//!    `validate_token`) in the top half of the seed pull in lines that share
//!    them: the sentence that answers rarely repeats the question's words.
//! 3. **Coverage.** The best-hitting lines from *every* turn that hits, so a
//!    narrow pick cannot drop a whole turn (the "fourth restaurant" failure).
//! 4. **Neighbours.** ±1 line from the same turn and speaker around each
//!    pick, because "…redeemed a coupon" depends on the line before it.
//! 5. **Budget.** Rendered in line order under a line cap and a byte cap.
//!
//! Deterministic: same store, same query, same bytes. The rendering sits at
//! the end of segment 5, so segments 1–4 stay byte-stable.

use crate::lines::{Line, LineStore, Speaker};
use std::collections::{BTreeMap, BTreeSet};

/// Retrieval limits. The note's defaults are 120 lines / ~7,500 tokens for a
/// chat memory; the harness is built for a short volatile tail, so its
/// default is a quarter of that and the metrics say whether to grow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrieveBudget {
    pub max_lines: usize,
    pub max_bytes: usize,
    /// Breadth of the deterministic pre-select.
    pub k_pre: usize,
    /// Neighbours re-attached on each side of a pick.
    pub neighbours: u32,
    /// Best lines kept per hitting turn for the coverage guarantee.
    pub per_turn: usize,
}

impl Default for RetrieveBudget {
    fn default() -> Self {
        Self {
            max_lines: 30,
            max_bytes: 6_000,
            k_pre: 60,
            neighbours: 1,
            per_turn: 3,
        }
    }
}

/// What retrieval returned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Retrieved {
    /// Line numbers in ascending order, the ones rendered.
    pub lines: Vec<u32>,
    /// Candidates that matched before the budget cut. When this exceeds
    /// `lines.len()` the budget was hit: the counter the compaction event
    /// waits on.
    pub candidates: usize,
    /// Every candidate the compactor has not judged yet, in line order: the
    /// input of a compaction event. Empty when nothing new needs judging.
    pub unjudged: Vec<u32>,
    /// The per-turn entity registry (`session_headers` in the prototype):
    /// one line per represented turn naming the entities the user mentioned
    /// there. Rendered ahead of the lines.
    pub headers: String,
    pub text: String,
}

impl Retrieved {
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn budget_hit(&self) -> bool {
        self.candidates > self.lines.len()
    }
}

const STOP: &[&str] = &[
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "be",
    "must",
    "may",
    "shall",
    "with",
    "by",
    "at",
    "from",
    "that",
    "this",
    "it",
    "as",
    "any",
    "all",
    "if",
    "then",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "under",
    "per",
    "did",
    "do",
    "does",
    "have",
    "has",
    "had",
    "you",
    "your",
    "me",
    "my",
    "i",
    "we",
    "our",
    "was",
    "were",
    "been",
    "can",
    "could",
    "would",
    "should",
    "about",
    "some",
    "there",
    "their",
    "so",
    "far",
    "we've",
    "established",
];

/// Light stemming: strip `ing`, `ed`, `es`, `s` when a stem of 3+ chars
/// remains (`kits` -> `kit`, `worked` -> `work`).
pub fn stem(w: &str) -> &str {
    for suf in ["ing", "ed", "es", "s"] {
        if w.len() >= suf.len() + 3 && w.ends_with(suf) {
            return &w[..w.len() - suf.len()];
        }
    }
    w
}

/// Content words of a text, lowercased, stemmed, stop words removed.
pub fn content_words(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/'))
        .flat_map(|tok| {
            // A path or dotted name counts both whole and by segment, so
            // `src/auth.rs` matches a query saying `auth.rs`.
            let whole = tok.trim_matches(|c: char| c == '.' || c == '/');
            let mut v = vec![whole.to_string()];
            v.extend(whole.split(['.', '/']).map(str::to_string));
            v
        })
        .filter(|w| w.len() > 2 && !STOP.contains(&w.as_str()))
        .map(|w| stem(&w).to_string())
        .collect()
}

/// Proper-noun-shaped terms: capitalised words not at a sentence start,
/// identifiers with `_`, paths and dotted names.
fn anchor_terms(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut first = true;
    for tok in text.split_whitespace() {
        let t =
            tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.' && c != '/');
        if t.is_empty() {
            continue;
        }
        let is_ident = t.contains('_') || t.contains('/') || (t.contains('.') && !t.ends_with('.'));
        let is_proper = !first && t.chars().next().is_some_and(char::is_uppercase) && t.len() >= 3;
        if is_ident || is_proper {
            out.insert(t.to_lowercase());
        }
        first = tok.ends_with('.') || tok.ends_with('?') || tok.ends_with('!');
    }
    out
}

impl LineStore {
    /// Retrieve the lines that matter for `query`, rendered.
    pub fn retrieve(&self, query: &str, budget: RetrieveBudget) -> Retrieved {
        if self.lines.is_empty() {
            return Retrieved::default();
        }
        let qw = content_words(query);
        if qw.is_empty() {
            return Retrieved::default();
        }

        // 1. term overlap
        let scored: Vec<(usize, &Line)> = self
            .lines
            .iter()
            .map(|l| (qw.intersection(&content_words(&l.text)).count(), l))
            .collect();
        let mut seed: Vec<&Line> = scored
            .iter()
            .filter(|(s, _)| *s > 0)
            .map(|(_, l)| *l)
            .collect();
        // Highest score first, then earliest line: stable.
        seed.sort_by(|a, b| {
            let sa = qw.intersection(&content_words(&a.text)).count();
            let sb = qw.intersection(&content_words(&b.text)).count();
            sb.cmp(&sa).then(a.n.cmp(&b.n))
        });
        seed.truncate(budget.k_pre);

        // 2. one-hop expansion through anchor terms of the top half
        let mut anchors: BTreeSet<String> = BTreeSet::new();
        for l in seed.iter().take(budget.k_pre.div_ceil(2)) {
            anchors.extend(anchor_terms(&l.text));
        }
        anchors.extend(anchor_terms(query));
        let seed_ids: BTreeSet<u32> = seed.iter().map(|l| l.n).collect();
        let hop: Vec<&Line> = self
            .lines
            .iter()
            .filter(|l| !seed_ids.contains(&l.n))
            .filter(|l| !anchors.is_empty() && !anchor_terms(&l.text).is_disjoint(&anchors))
            .collect();

        let mut chosen: BTreeSet<u32> = seed_ids.clone();
        for l in hop.iter().take(budget.k_pre.saturating_sub(seed.len())) {
            chosen.insert(l.n);
        }

        // 3. coverage: the best lines of every turn that hits
        let mut best_by_turn: BTreeMap<u32, Vec<(usize, u32)>> = BTreeMap::new();
        for (s, l) in &scored {
            if *s >= 2 || (*s >= 1 && qw.len() <= 3) {
                best_by_turn.entry(l.turn).or_default().push((*s, l.n));
            }
        }
        let mut coverage: BTreeSet<u32> = BTreeSet::new();
        for hits in best_by_turn.values_mut() {
            hits.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            for (_, n) in hits.iter().take(budget.per_turn) {
                chosen.insert(*n);
                coverage.insert(*n);
            }
        }

        // 4. neighbours, same turn and speaker
        let mut with_neighbours = chosen.clone();
        for n in &chosen {
            let Some(base) = self.get(*n) else { continue };
            for d in 1..=budget.neighbours {
                for m in [n.checked_sub(d), n.checked_add(d)].into_iter().flatten() {
                    if let Some(nb) = self.get(m)
                        && nb.turn == base.turn
                        && nb.speaker == base.speaker
                    {
                        with_neighbours.insert(m);
                    }
                }
            }
        }

        // A compaction event's verdicts apply here: a judged line that was
        // not kept is out. Unjudged candidates are reported so the event can
        // run on exactly them, never re-judging. Coverage picks bypass the
        // verdict, as the prototype's `kept | coverage_hits` does: the
        // narrower may drop a whole turn, coverage puts its best lines back.
        let unjudged: Vec<u32> = with_neighbours
            .iter()
            .copied()
            .filter(|n| !self.judged.contains(n))
            .collect();
        with_neighbours.retain(|n| self.admits(*n) || coverage.contains(n));

        // 5. budget, in line order. When the cut lands, keep the *latest*
        // lines: a knowledge-update question needs the newest statement, and
        // a blind slice by number would drop the newest turns first.
        let candidates = with_neighbours.len();
        let mut ordered: Vec<u32> = with_neighbours.into_iter().collect();
        if ordered.len() > budget.max_lines {
            ordered = ordered[ordered.len() - budget.max_lines..].to_vec();
        }
        let mut text = String::new();
        let mut kept = Vec::new();
        for n in ordered {
            let Some(l) = self.get(n) else { continue };
            let line = l.render();
            if !text.is_empty() && text.len() + line.len() + 1 > budget.max_bytes {
                break;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&line);
            kept.push(n);
        }
        let turns: BTreeSet<u32> = kept.iter().filter_map(|n| self.get(*n)).map(|l| l.turn).collect();
        let headers = self.turn_headers(&turns);
        Retrieved {
            lines: kept,
            candidates,
            unjudged,
            headers,
            text,
        }
    }

    /// `session_headers`: one deterministic line per represented turn with
    /// the entities the USER mentioned there -- capitalised tokens not at a
    /// sentence start, the top eight by frequency -- so "where" context
    /// survives selection without the full text.
    pub fn turn_headers(&self, turns: &BTreeSet<u32>) -> String {
        let mut out = Vec::new();
        for t in turns {
            let mut freq: BTreeMap<String, usize> = BTreeMap::new();
            for l in self.lines.iter().filter(|l| l.turn == *t && l.speaker == Speaker::User) {
                for ent in user_entities(&l.text) {
                    *freq.entry(ent).or_default() += 1;
                }
            }
            let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let ents: Vec<String> = ranked.into_iter().take(8).map(|(w, _)| w).collect();
            if !ents.is_empty() {
                out.push(format!("— turn {t} user mentions: {}", ents.join(", ")));
            }
        }
        out.join("\n")
    }
}

/// Capitalised words (3+ letters, optionally a capitalised pair) that do not
/// open a sentence, minus stop words and determiners: the prototype's
/// `session_headers` regex.
fn user_entities(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let toks: Vec<&str> = text.split_whitespace().collect();
    let mut sentence_start = true;
    let mut i = 0;
    while i < toks.len() {
        let raw = toks[i];
        let w = raw.trim_matches(|c: char| !c.is_alphabetic());
        let is_cap = w.len() >= 3
            && w.chars().next().is_some_and(char::is_uppercase)
            && w.chars().all(char::is_alphabetic);
        if is_cap && !sentence_start {
            let lower = w.to_lowercase();
            if !STOP.contains(&lower.as_str())
                && !["The", "This", "That", "What", "When", "Where"].contains(&w)
            {
                // Pair with a following capitalised word, as the regex does.
                let mut name = w.to_string();
                if !raw.ends_with(['.', '!', '?', ','])
                    && let Some(next) = toks.get(i + 1)
                {
                    let nw = next.trim_matches(|c: char| !c.is_alphabetic());
                    if nw.len() >= 3
                        && nw.chars().next().is_some_and(char::is_uppercase)
                        && nw.chars().all(char::is_alphabetic)
                    {
                        name.push(' ');
                        name.push_str(nw);
                        i += 1;
                    }
                }
                out.push(name);
            }
        }
        sentence_start = raw.ends_with(['.', '!', '?']);
        i += 1;
    }
    out
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn turn_headers_list_user_entities_not_sentence_openers() {
        let mut s = LineStore::new();
        s.add(1, Speaker::User, "Costco sells the kits. We bought them at Costco with Maria Lopez.", 1000);
        s.add(1, Speaker::Assistant, "Costco Costco Costco.", 1000);
        s.add(2, Speaker::User, "Nothing capitalised here.", 1000);
        let h = s.turn_headers(&[1, 2].into_iter().collect());
        assert_eq!(h, "— turn 1 user mentions: Costco, Maria Lopez");
    }
}

impl LineStore {
    /// Render a set of lines in order, for the compactor's input.
    pub fn render_lines(&self, ns: &[u32]) -> String {
        ns.iter()
            .filter_map(|n| self.get(*n))
            .map(Line::render)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The lines of the last `turns` turns, verbatim, for the working window
    /// (the note's "last two turns"). Speaker filter optional.
    pub fn recent(&self, turns: u32, speaker: Option<Speaker>) -> Vec<&Line> {
        let Some(last) = self.lines.last().map(|l| l.turn) else {
            return Vec::new();
        };
        let floor = last.saturating_sub(turns.saturating_sub(1));
        self.lines
            .iter()
            .filter(|l| l.turn >= floor && speaker.is_none_or(|s| l.speaker == s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> LineStore {
        let mut s = LineStore::new();
        s.add(
            1,
            Speaker::User,
            "I switched from Target to Costco for groceries last month.",
            500,
        );
        s.add(
            1,
            Speaker::Assistant,
            "Costco's Kirkland brand covers most of your list at lower unit cost.",
            500,
        );
        s.add(
            2,
            Speaker::User,
            "The auth module in src/auth.rs depends on the jsonwebtoken crate.",
            500,
        );
        s.add(
            2,
            Speaker::Tool,
            "```\nfn validate_token(t: &str) -> bool { todo!() }\n```",
            500,
        );
        s.add(
            3,
            Speaker::User,
            "Remind me where I shop for groceries now.",
            500,
        );
        s.add(4, Speaker::Assistant, "The weather in Paris is mild.", 500);
        s
    }

    #[test]
    fn term_overlap_finds_the_answer_bearing_line() {
        let r = store().retrieve("Where do I buy groceries?", RetrieveBudget::default());
        assert!(r.lines.contains(&1), "{r:?}");
        assert!(r.text.contains("Costco"));
        assert!(!r.text.contains("Paris"));
    }

    #[test]
    fn one_hop_expansion_reaches_a_line_that_shares_only_an_anchor() {
        // The query names Costco; the Kirkland line never says "groceries"
        // but shares the anchor "Costco's".
        let r = store().retrieve("What did you say about Costco?", RetrieveBudget::default());
        assert!(r.lines.contains(&2), "{r:?}");
    }

    #[test]
    fn paths_match_by_segment_and_blocks_come_back_whole() {
        let r = store().retrieve("What is in auth.rs?", RetrieveBudget::default());
        assert!(r.lines.contains(&3), "{r:?}");
        let r = store().retrieve("show validate_token", RetrieveBudget::default());
        assert!(r.text.contains("fn validate_token"), "{r:?}");
    }

    #[test]
    fn retrieval_is_deterministic_and_ordered() {
        let s = store();
        let a = s.retrieve("groceries at Costco", RetrieveBudget::default());
        let b = s.retrieve("groceries at Costco", RetrieveBudget::default());
        assert_eq!(a, b);
        assert!(a.lines.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn the_budget_keeps_the_latest_lines_and_reports_the_cut() {
        let mut s = LineStore::new();
        for t in 1..=40 {
            s.add(
                t,
                Speaker::User,
                &format!("Note {t}: the cache page is 256 tokens."),
                200,
            );
        }
        let r = s.retrieve(
            "cache page tokens",
            RetrieveBudget {
                max_lines: 5,
                ..RetrieveBudget::default()
            },
        );
        assert_eq!(r.lines.len(), 5);
        assert!(r.budget_hit());
        assert_eq!(*r.lines.last().unwrap(), 40, "latest line survives the cut");
    }

    #[test]
    fn a_compaction_verdict_hides_judged_lines_that_were_not_kept() {
        let mut s = LineStore::new();
        for t in 1..=3 {
            // Two equally matching lines per turn; coverage keeps one per
            // turn (the earliest), the other is subject to the verdict.
            s.add(
                t,
                Speaker::User,
                &format!("Note {t}: the cache page is 256 tokens. Also the cache page tokens matter here."),
                200,
            );
        }
        let budget = RetrieveBudget {
            per_turn: 1,
            neighbours: 0,
            ..RetrieveBudget::default()
        };
        let before = s.retrieve("cache page tokens", budget);
        assert_eq!(before.lines.len(), 6);
        assert_eq!(before.unjudged.len(), 6, "nothing judged yet");

        // The compactor judged lines 1-4 and kept only 4. Line 1 and 3 are
        // coverage picks, so they come back regardless (`kept | coverage`);
        // line 2 was judged and dropped, so it is out.
        s.record_compaction(&[1, 2, 3, 4], &[4]);
        let after = s.retrieve("cache page tokens", budget);
        assert_eq!(after.lines, vec![1, 3, 4, 5, 6]);
        assert_eq!(
            after.unjudged,
            vec![5, 6],
            "only the new lines await judgement"
        );
        assert_eq!(s.render_lines(&[2]).lines().count(), 1);
    }

    #[test]
    fn an_empty_store_or_query_returns_nothing() {
        assert!(
            LineStore::new()
                .retrieve("x", RetrieveBudget::default())
                .is_empty()
        );
        assert!(
            store()
                .retrieve("the of and", RetrieveBudget::default())
                .is_empty()
        );
    }

    #[test]
    fn recent_returns_the_last_turns_verbatim() {
        let s = store();
        let last2 = s.recent(2, None);
        assert!(last2.iter().all(|l| l.turn >= 3));
        assert_eq!(last2.len(), 2);
    }
}
