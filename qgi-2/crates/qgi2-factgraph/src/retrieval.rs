//! Entry-point retrieval.
//!
//! Spec:
//!
//! > Embedder — Qwen3-Embedding-0.6B via vLLM `/v1/embeddings`
//! > (MiniLM fallback) — **entry-point retrieval only**.
//!
//! The embedder's job is narrow on purpose: it picks which nodes to *start*
//! from, and the mood's traversal ([`crate::traversal`]) does the rest. That is
//! what keeps retrieval explainable — a rule-driven walk from a small set of
//! seeds — rather than a similarity search whose results nobody can account
//! for.
//!
//! Under the Quick profile ([`RetrievalPolicy::lexical_only`]) no embedding
//! call is made at all: exact-key and lexical matching pick the seeds.

use crate::store::FactGraph;
use qgi2_spec_types::RetrievalPolicy;
use std::collections::BTreeMap;

/// A node chosen to start traversal from.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryPoint {
    pub node: String,
    /// Similarity or lexical score in `[0, 1]`. Used for ranking only; it never
    /// reaches the rendered prompt, so it cannot perturb the cache.
    pub score: f32,
    pub how: EntryMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryMethod {
    /// The query mentioned this node verbatim.
    ExactKey,
    /// Token overlap between the query and the node string.
    Lexical,
    /// Cosine similarity against a stored embedding.
    Embedding,
}

/// Picks entry points for a turn.
pub struct Retrieval {
    /// node -> embedding, populated by the embedder. Absent under Quick.
    embeddings: BTreeMap<String, Vec<f32>>,
    max_entries: usize,
}

impl Default for Retrieval {
    fn default() -> Self {
        Self::new(8)
    }
}

impl Retrieval {
    pub fn new(max_entries: usize) -> Self {
        Self {
            embeddings: BTreeMap::new(),
            max_entries,
        }
    }

    /// Store an embedding for a node.
    pub fn set_embedding(&mut self, node: impl Into<String>, vector: Vec<f32>) {
        self.embeddings.insert(node.into(), vector);
    }

    pub fn has_embedding(&self, node: &str) -> bool {
        self.embeddings.contains_key(node)
    }

    /// Nodes in the graph that still need an embedding.
    pub fn missing_embeddings(&self, graph: &FactGraph) -> Vec<String> {
        graph
            .subjects()
            .filter(|s| !self.embeddings.contains_key(*s))
            .map(|s| s.to_string())
            .collect()
    }

    /// Choose entry points for `query`.
    ///
    /// Exact-key matches always rank first: if the user named a node, that node
    /// is the entry point, and no embedding score should outrank it.
    /// `query_embedding` is `None` under the Quick profile, and the method
    /// degrades to lexical rather than returning nothing.
    pub fn entry_points(
        &self,
        graph: &FactGraph,
        query: &str,
        query_embedding: Option<&[f32]>,
        policy: RetrievalPolicy,
    ) -> Vec<EntryPoint> {
        let q_lower = query.to_lowercase();
        let q_tokens = tokenize(&q_lower);
        let mut scored: BTreeMap<String, EntryPoint> = BTreeMap::new();

        for node in graph.subjects() {
            let node_lower = node.to_lowercase();

            if q_lower.contains(&node_lower) {
                scored.insert(
                    node.to_string(),
                    EntryPoint {
                        node: node.to_string(),
                        score: 1.0,
                        how: EntryMethod::ExactKey,
                    },
                );
                continue;
            }

            let lex = lexical_score(&q_tokens, &node_lower);
            if lex > 0.0 {
                scored.insert(
                    node.to_string(),
                    EntryPoint {
                        node: node.to_string(),
                        score: lex,
                        how: EntryMethod::Lexical,
                    },
                );
            }
        }

        if !policy.lexical_only
            && let Some(qv) = query_embedding
        {
            for (node, vector) in &self.embeddings {
                let sim = cosine(qv, vector);
                if sim <= 0.0 {
                    continue;
                }
                // Never downgrade an exact-key hit to an embedding hit.
                match scored.get(node) {
                    Some(existing) if existing.how == EntryMethod::ExactKey => continue,
                    Some(existing) if existing.score >= sim => continue,
                    _ => {}
                }
                scored.insert(
                    node.clone(),
                    EntryPoint {
                        node: node.clone(),
                        score: sim,
                        how: EntryMethod::Embedding,
                    },
                );
            }
        }

        let mut out: Vec<EntryPoint> = scored.into_values().collect();
        // Sort by score descending, then by node name, so equal scores have a
        // deterministic order rather than depending on map iteration.
        out.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.node.cmp(&b.node))
        });
        out.truncate(self.max_entries);
        out
    }
}

fn tokenize(s: &str) -> Vec<&str> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .collect()
}

/// Fraction of the node's tokens that the query also contains.
fn lexical_score(q_tokens: &[&str], node: &str) -> f32 {
    let n_tokens = tokenize(node);
    if n_tokens.is_empty() {
        return 0.0;
    }
    let hits = n_tokens.iter().filter(|t| q_tokens.contains(t)).count();
    hits as f32 / n_tokens.len() as f32
}

/// Cosine similarity. Returns 0 for a zero vector or a length mismatch rather
/// than NaN, so a malformed embedding degrades to "no signal" instead of
/// poisoning the ranking sort.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Scope;
    use qgi2_spec_types::{
        CommitToken, Confidence, ConflictPolicy, Profile, ProposedFact, Relation, Source,
    };

    fn graph() -> FactGraph {
        let mut g = FactGraph::new();
        for (s, o) in [
            ("task:authentication", "file:auth.rs"),
            ("task:database", "file:db.rs"),
            ("person:sam", "topic:rust"),
        ] {
            let f = ProposedFact {
                subject: s.into(),
                relation: Relation::DependsOn,
                object: o.into(),
                confidence: Confidence::new(0.9),
                evidence: None,
            }
            .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
            g.commit(f, Scope::Session, ConflictPolicy::KeepBoth);
        }
        g
    }

    #[test]
    fn exact_key_matches_win() {
        let g = graph();
        let r = Retrieval::default();
        let entries = r.entry_points(
            &g,
            "what does task:authentication need?",
            None,
            Profile::Traceable.retrieval(),
        );
        assert_eq!(entries[0].node, "task:authentication");
        assert_eq!(entries[0].how, EntryMethod::ExactKey);
    }

    #[test]
    fn lexical_matching_works_without_embeddings() {
        let g = graph();
        let r = Retrieval::default();
        let entries = r.entry_points(&g, "tell me about the database", None, Profile::Quick.retrieval());
        assert!(entries.iter().any(|e| e.node == "task:database"));
    }

    #[test]
    fn quick_profile_ignores_embeddings_entirely() {
        let g = graph();
        let mut r = Retrieval::default();
        r.set_embedding("person:sam", vec![1.0, 0.0]);
        let entries = r.entry_points(&g, "unrelated words", Some(&[1.0, 0.0]), Profile::Quick.retrieval());
        assert!(
            !entries.iter().any(|e| e.how == EntryMethod::Embedding),
            "Quick must not use the embedder: {entries:?}"
        );
    }

    #[test]
    fn embeddings_contribute_when_the_profile_allows() {
        let g = graph();
        let mut r = Retrieval::default();
        r.set_embedding("person:sam", vec![1.0, 0.0]);
        let entries = r.entry_points(&g, "unrelated words", Some(&[1.0, 0.0]), Profile::Traceable.retrieval());
        assert!(entries.iter().any(|e| e.node == "person:sam" && e.how == EntryMethod::Embedding));
    }

    #[test]
    fn an_embedding_never_displaces_an_exact_key_hit() {
        let g = graph();
        let mut r = Retrieval::default();
        r.set_embedding("task:database", vec![1.0, 0.0]);
        let entries = r.entry_points(
            &g,
            "task:database please",
            Some(&[1.0, 0.0]),
            Profile::Traceable.retrieval(),
        );
        let e = entries.iter().find(|e| e.node == "task:database").unwrap();
        assert_eq!(e.how, EntryMethod::ExactKey);
    }

    #[test]
    fn malformed_embeddings_score_zero_rather_than_nan() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn ranking_is_deterministic_for_equal_scores() {
        let g = graph();
        let r = Retrieval::default();
        let a = r.entry_points(&g, "task", None, Profile::Traceable.retrieval());
        let b = r.entry_points(&g, "task", None, Profile::Traceable.retrieval());
        assert_eq!(a, b);
    }

    #[test]
    fn missing_embeddings_lists_unembedded_subjects() {
        let g = graph();
        let mut r = Retrieval::default();
        r.set_embedding("person:sam", vec![1.0, 0.0]);
        let missing = r.missing_embeddings(&g);
        assert!(missing.contains(&"task:database".to_string()));
        assert!(!missing.contains(&"person:sam".to_string()));
    }
}
