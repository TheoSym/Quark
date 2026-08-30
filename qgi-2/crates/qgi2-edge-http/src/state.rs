//! Shared server state: the session store.

use anyhow::Result;
use qgi2_engine_vllm::EngineRegistry;
use qgi2_factgraph::FactGraph;
use qgi2_spec_types::Persona;
use qgi2_turn::{Session, SessionConfig, session::SkillCandidate};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Sessions, keyed by persona.
///
/// Keying by persona rather than by connection is deliberate. A session owns
/// the assembler's memory of the stable prefix, and that memory is only
/// meaningful for one persona: the mood segment differs between them, so a
/// Builder request and a Researcher request cannot share a cached prefix
/// anyway. Two clients on the same persona *should* share, because they share
/// the prefix.
pub struct SessionStore {
    sessions: Mutex<HashMap<String, Arc<Mutex<Session>>>>,
    config: SessionConfig,
    registry: EngineRegistry,
    skills: Vec<SkillCandidate>,
    /// Where graphs are persisted between runs.
    graph_dir: Option<PathBuf>,
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
            config,
            registry,
            skills,
            graph_dir,
        }
    }

    fn key(persona: Persona) -> String {
        format!("{}-{}", persona.mood.as_str(), persona.profile.as_str())
    }

    fn graph_path(&self, persona: Persona) -> Option<PathBuf> {
        self.graph_dir
            .as_ref()
            .map(|d| d.join(format!("{}.json", Self::key(persona))))
    }

    /// The session for a persona, creating and loading it on first use.
    pub async fn get(&self, persona: Persona) -> Arc<Mutex<Session>> {
        let key = Self::key(persona);
        let mut map = self.sessions.lock().await;
        if let Some(s) = map.get(&key) {
            return s.clone();
        }

        let config = SessionConfig {
            persona,
            ..self.config.clone()
        };
        let mut session = Session::new(config, self.registry.clone(), self.skills.clone());

        // A durable slice that fails to load is a cold graph, not a failed
        // request: the agent works, it just does not remember. Refusing to
        // serve would be a worse trade.
        if let Some(path) = self.graph_path(persona)
            && let Ok(text) = std::fs::read_to_string(&path)
        {
            match FactGraph::from_json(&text) {
                Ok(g) => session = session.with_graph(g),
                Err(e) => tracing::warn!(?path, error = %e, "ignoring unreadable fact graph"),
            }
        }

        let arc = Arc::new(Mutex::new(session));
        map.insert(key, arc.clone());
        arc
    }

    /// Persist every live session's graph.
    pub async fn persist_all(&self) -> Result<usize> {
        let Some(dir) = &self.graph_dir else {
            return Ok(0);
        };
        std::fs::create_dir_all(dir)?;

        let map = self.sessions.lock().await;
        let mut written = 0;
        for (key, session) in map.iter() {
            let session = session.lock().await;
            let json = session.graph_json()?;
            let path = dir.join(format!("{key}.json"));
            // Write to a temp file and rename: a crash partway through a direct
            // write leaves a truncated graph that the next run silently loads
            // as a smaller memory.
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, json)?;
            std::fs::rename(&tmp, &path)?;
            written += 1;
        }
        Ok(written)
    }

    /// End every session: promote, decay, write metric facts, then persist.
    pub async fn end_all(&self) -> Result<usize> {
        {
            let map = self.sessions.lock().await;
            for session in map.values() {
                session.lock().await.end_session();
            }
        }
        self.persist_all().await
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
    use qgi2_engine_vllm::Endpoint;
    use qgi2_spec_types::{ModelRole, Mood, Profile, Speculation};

    fn store() -> SessionStore {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://127.0.0.1:8000/v1", "p", Speculation::Mtp { n: 2 }),
        );
        SessionStore::new(SessionConfig::default(), r, vec![], None)
    }

    #[tokio::test]
    async fn the_same_persona_shares_one_session() {
        let s = store();
        let a = s.get(Persona::default()).await;
        let b = s.get(Persona::default()).await;
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn different_personas_get_different_sessions() {
        // They cannot share a cached prefix, because the mood segment differs.
        let s = store();
        let a = s.get(Persona::new(Mood::Builder, Profile::Traceable)).await;
        let b = s.get(Persona::new(Mood::Researcher, Profile::Traceable)).await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn persisting_without_a_directory_is_a_no_op() {
        assert_eq!(store().persist_all().await.unwrap(), 0);
    }
}
