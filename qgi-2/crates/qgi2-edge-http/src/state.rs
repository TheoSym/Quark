//! Shared server state: the session store.

use anyhow::Result;
use qgi2_engine::EngineRegistry;
use qgi2_factgraph::{FactGraph, Retrieval, Scope};
use qgi2_spec_types::Persona;
use qgi2_turn::{Session, SessionConfig, session::SkillCandidate};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Sessions, keyed by persona and client.
///
/// # Keys
///
/// A session owns the assembler's memory of the stable prefix and the facts
/// learned this session, and a mutex around it. Keying by persona alone made
/// every jcode client on `builder-traceable` serialize behind one lock, so one
/// slow turn stalled everyone. The key is now `persona` plus a client id, taken
/// from the `X-QGI2-Session` header or the OpenAI `user` field; a client that
/// sends neither falls back to the persona-wide session, which is the old
/// behaviour and fine for a single user.
///
/// # Files
///
/// - `<persona>.durable.json` -- the durable slice, shared across clients of a
///   persona and merged into on session end. This is what segment 3 renders.
/// - `<key>.session.json` -- the full graph of one live session, rewritten
///   after every turn. Crash insurance: a server that dies mid-session loses at
///   most the turn in flight, and the next request for that key resumes from
///   it.
/// - `<key>.embeddings.json` -- node vectors, alongside the session, so a
///   restart does not re-embed every subject.
///
/// # Ending
///
/// A server serving many conversations has no natural "session end", which is
/// where promotion, decay and the metric facts were supposed to run. They now
/// run when a session goes idle for [`SessionStore::idle_timeout`], when a
/// client asks via `/qgi2/end`, or at shutdown -- whichever comes first.
pub struct SessionStore {
    sessions: Mutex<HashMap<String, Arc<Mutex<Session>>>>,
    last_active: Mutex<HashMap<String, Instant>>,
    config: SessionConfig,
    registry: EngineRegistry,
    skills: Vec<SkillCandidate>,
    /// Where graphs are persisted between runs.
    graph_dir: Option<PathBuf>,
    idle_timeout: Duration,
}

impl SessionStore {
    pub fn new(
        config: SessionConfig,
        registry: EngineRegistry,
        skills: Vec<SkillCandidate>,
        graph_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            last_active: Mutex::new(HashMap::new()),
            config,
            registry,
            skills,
            graph_dir,
            idle_timeout: Duration::from_secs(30 * 60),
        }
    }

    pub fn with_idle_timeout(mut self, idle: Duration) -> Self {
        self.idle_timeout = idle;
        self
    }

    pub fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }

    fn persona_key(persona: Persona) -> String {
        format!("{}-{}", persona.mood.as_str(), persona.profile.as_str())
    }

    /// The session key for a persona and optional client id.
    pub fn key(persona: Persona, client: Option<&str>) -> String {
        match client.map(str::trim).filter(|c| !c.is_empty()) {
            // Sanitise: the key becomes a filename.
            Some(c) => format!(
                "{}@{}",
                Self::persona_key(persona),
                c.chars()
                    .map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
                    .take(64)
                    .collect::<String>()
            ),
            None => Self::persona_key(persona),
        }
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        self.graph_dir.as_ref().map(|d| d.join(name))
    }

    fn durable_path(&self, persona: Persona) -> Option<PathBuf> {
        self.path(&format!("{}.durable.json", Self::persona_key(persona)))
    }

    fn session_path(&self, key: &str) -> Option<PathBuf> {
        self.path(&format!("{key}.session.json"))
    }

    fn embeddings_path(&self, key: &str) -> Option<PathBuf> {
        self.path(&format!("{key}.embeddings.json"))
    }

    /// Atomic write: a crash partway through a direct write leaves a truncated
    /// file that the next run silently loads as a smaller memory.
    fn write_atomic(path: &PathBuf, contents: &str) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, contents)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn load_graph(path: Option<PathBuf>) -> Option<FactGraph> {
        let path = path?;
        let text = std::fs::read_to_string(&path).ok()?;
        match FactGraph::from_json(&text) {
            Ok(g) => Some(g),
            Err(e) => {
                // An unreadable graph is a cold start, not a failed request:
                // the agent works, it just does not remember.
                tracing::warn!(?path, error = %e, "ignoring unreadable fact graph");
                None
            }
        }
    }

    /// The session for a persona and client, creating and loading it on first
    /// use.
    pub async fn get(&self, persona: Persona, client: Option<&str>) -> Arc<Mutex<Session>> {
        let key = Self::key(persona, client);
        self.last_active.lock().await.insert(key.clone(), Instant::now());

        let mut map = self.sessions.lock().await;
        if let Some(s) = map.get(&key) {
            return s.clone();
        }

        let config = SessionConfig {
            persona,
            ..self.config.clone()
        };
        let mut session = Session::new(config, self.registry.clone(), self.skills.clone());

        // A live session file means the server died mid-session; resume from
        // it. Otherwise start from the persona's shared durable slice.
        if let Some(g) = Self::load_graph(self.session_path(&key)) {
            tracing::info!(%key, facts = g.len(), "resuming session from disk");
            session = session.with_graph(g);
        } else if let Some(g) = Self::load_graph(self.durable_path(persona)) {
            session = session.with_graph(g);
        }

        if let Some(path) = self.embeddings_path(&key)
            && let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(r) = Retrieval::from_json(&text)
        {
            session = session.with_retrieval(r);
        }

        let arc = Arc::new(Mutex::new(session));
        map.insert(key, arc.clone());
        arc
    }

    /// Record that a session was just used.
    pub async fn touch(&self, key: &str) {
        self.last_active
            .lock()
            .await
            .insert(key.to_string(), Instant::now());
    }

    /// Write one session's live state. Called after every turn.
    pub async fn persist_session(&self, key: &str) -> Result<()> {
        let session = {
            let map = self.sessions.lock().await;
            let Some(s) = map.get(key) else { return Ok(()) };
            s.clone()
        };
        let (graph, retrieval) = {
            let s = session.lock().await;
            (s.graph_json()?, s.retrieval_json()?)
        };
        if let Some(p) = self.session_path(key) {
            Self::write_atomic(&p, &graph)?;
        }
        if let Some(p) = self.embeddings_path(key) {
            Self::write_atomic(&p, &retrieval)?;
        }
        Ok(())
    }

    /// End one session: promote, decay, write metric facts, merge its durable
    /// slice into the persona's shared file, and forget it.
    ///
    /// Merge rather than overwrite: two clients of one persona ending in either
    /// order must both contribute, and `FactGraph::commit` already knows how
    /// to reinforce a fact the other session also learned.
    pub async fn end(&self, key: &str) -> Result<bool> {
        let session = {
            let mut map = self.sessions.lock().await;
            let Some(s) = map.remove(key) else { return Ok(false) };
            s
        };
        self.last_active.lock().await.remove(key);

        let mut s = session.lock().await;
        let persona = s.persona();
        let summary = s.end_session();
        tracing::info!(%key, promoted = summary.promoted, dropped = summary.dropped, "session ended");

        if let Some(durable_path) = self.durable_path(persona) {
            let mut shared = Self::load_graph(Some(durable_path.clone())).unwrap_or_default();
            let policy = persona.mood.table().conflict;
            for f in s.graph.iter_scope(Scope::Durable) {
                shared.commit(f.clone(), Scope::Durable, policy);
            }
            Self::write_atomic(&durable_path, &shared.to_json()?)?;
        }

        for p in [self.session_path(key), self.embeddings_path(key)].into_iter().flatten() {
            let _ = std::fs::remove_file(p);
        }
        Ok(true)
    }

    /// End every session idle longer than the timeout. Run periodically.
    pub async fn sweep(&self) -> Result<usize> {
        let cutoff = Instant::now() - self.idle_timeout;
        let idle: Vec<String> = self
            .last_active
            .lock()
            .await
            .iter()
            .filter(|(_, t)| **t < cutoff)
            .map(|(k, _)| k.clone())
            .collect();
        let mut ended = 0;
        for key in idle {
            if self.end(&key).await? {
                ended += 1;
            }
        }
        Ok(ended)
    }

    /// End every session. Used at shutdown.
    pub async fn end_all(&self) -> Result<usize> {
        let keys: Vec<String> = self.sessions.lock().await.keys().cloned().collect();
        let mut ended = 0;
        for key in keys {
            if self.end(&key).await? {
                ended += 1;
            }
        }
        Ok(ended)
    }

    /// Persist every live session without ending it.
    pub async fn persist_all(&self) -> Result<usize> {
        let keys: Vec<String> = self.sessions.lock().await.keys().cloned().collect();
        for key in &keys {
            self.persist_session(key).await?;
        }
        Ok(keys.len())
    }

    /// A live session by key, without creating one.
    pub async fn peek(&self, key: &str) -> Option<Arc<Mutex<Session>>> {
        self.sessions.lock().await.get(key).cloned()
    }

    pub async fn live_keys(&self) -> Vec<String> {
        self.sessions.lock().await.keys().cloned().collect()
    }
}


/// Axum state.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<SessionStore>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_engine::Endpoint;
    use qgi2_spec_types::{ConflictPolicy, ModelRole, Mood, Profile, Speculation};

    fn store() -> SessionStore {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://127.0.0.1:8000/v1", "p", Speculation::Mtp { n: 2 }),
        );
        SessionStore::new(SessionConfig::default(), r, vec![], None)
    }

    #[tokio::test]
    async fn the_same_persona_and_client_share_one_session() {
        let s = store();
        let a = s.get(Persona::default(), Some("sam")).await;
        let b = s.get(Persona::default(), Some("sam")).await;
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn different_clients_on_one_persona_do_not_share_a_lock() {
        // One slow turn must not stall every user of builder-traceable.
        let s = store();
        let a = s.get(Persona::default(), Some("sam")).await;
        let b = s.get(Persona::default(), Some("alice")).await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn no_client_id_falls_back_to_the_persona_wide_session() {
        let s = store();
        let a = s.get(Persona::default(), None).await;
        let b = s.get(Persona::default(), Some("")).await;
        assert!(Arc::ptr_eq(&a, &b), "empty and absent are the same client");
    }

    #[tokio::test]
    async fn different_personas_get_different_sessions() {
        // They cannot share a cached prefix, because the mood segment differs.
        let s = store();
        let a = s.get(Persona::new(Mood::Builder, Profile::Traceable), None).await;
        let b = s.get(Persona::new(Mood::Researcher, Profile::Traceable), None).await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn client_ids_are_sanitised_into_filenames() {
        let k = SessionStore::key(Persona::default(), Some("../etc/passwd"));
        assert!(!k.contains('/'), "{k}");
        assert!(!k.contains(".."), "{k}");
    }

    #[tokio::test]
    async fn ending_a_session_forgets_it() {
        let s = store();
        let key = SessionStore::key(Persona::default(), Some("sam"));
        s.get(Persona::default(), Some("sam")).await;
        assert!(s.end(&key).await.unwrap());
        assert!(!s.end(&key).await.unwrap(), "already gone");
        assert!(s.live_keys().await.is_empty());
    }

    #[tokio::test]
    async fn sweep_ends_only_idle_sessions() {
        let s = store().with_idle_timeout(Duration::from_millis(50));
        s.get(Persona::default(), Some("old")).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        s.get(Persona::default(), Some("fresh")).await;
        assert_eq!(s.sweep().await.unwrap(), 1);
        let live = s.live_keys().await;
        assert_eq!(live.len(), 1);
        assert!(live[0].ends_with("@fresh"));
    }

    #[tokio::test]
    async fn persisting_without_a_directory_is_a_no_op() {
        let s = store();
        s.get(Persona::default(), None).await;
        // No graph_dir: nothing written, nothing fails.
        assert!(s.persist_all().await.is_ok());
    }

    #[tokio::test]
    async fn a_crashed_session_resumes_from_its_file() {
        use qgi2_spec_types::{CommitToken, Confidence, ProposedFact, Relation, Source};
        let dir = std::env::temp_dir().join(format!("qgi2-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://127.0.0.1:8000/v1", "p", Speculation::Mtp { n: 2 }),
        );
        let s = SessionStore::new(SessionConfig::default(), r.clone(), vec![], Some(dir.clone()));
        let key = SessionStore::key(Persona::default(), Some("c"));
        {
            let sess = s.get(Persona::default(), Some("c")).await;
            let f = ProposedFact {
                subject: "task:a".into(),
                relation: Relation::DependsOn,
                object: "file:x".into(),
                confidence: Confidence::new(0.9),
                evidence: None,
            }
            .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
            sess.lock().await.graph.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        }
        s.persist_session(&key).await.unwrap();

        // A "new server": fresh store, same directory, same key.
        let s2 = SessionStore::new(SessionConfig::default(), r, vec![], Some(dir.clone()));
        let sess = s2.get(Persona::default(), Some("c")).await;
        assert_eq!(sess.lock().await.graph.len(), 1, "session facts survived the crash");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
