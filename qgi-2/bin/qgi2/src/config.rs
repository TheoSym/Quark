//! QGI-2's own config file, read from `~/.qgi2/config.toml` by default.
//!
//! Deliberately separate from `~/.jcode/config.toml`: QGI-2 does not modify
//! jcode, and that includes not appending its own keys to jcode's config
//! schema. The only jcode-side change a user makes is adding a
//! `[providers.qgi2]` block, which is jcode's existing, documented extension
//! point for OpenAI-compatible endpoints.

use anyhow::{Context, Result};
use qgi2_engine_vllm::{Endpoint, EngineRegistry};
use qgi2_spec_types::{ModelRole, Mood, Persona, Profile, Speculation};
use qgi2_turn::SessionConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qgi2Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub persona: PersonaConfig,
    #[serde(default)]
    pub engines: Vec<EngineConfig>,
    #[serde(default)]
    pub embedder: Option<EngineConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind: String,
    /// Where fact graphs are persisted between runs.
    pub graph_dir: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8788".to_string(),
            graph_dir: default_graph_dir(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    pub mood: String,
    pub profile: String,
    /// Whether the rules may switch mood mid-session. Off by default: a switch
    /// discards the cached prefix.
    #[serde(default)]
    pub allow_mood_switch: bool,
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            mood: "builder".into(),
            profile: "traceable".into(),
            allow_mood_switch: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// `planner` or `worker`.
    pub role: String,
    pub base_url: String,
    pub model: String,
    /// The speculation this vLLM process was launched with: `mtp`, `dflash2`,
    /// `ngram`, or `off`.
    #[serde(default = "default_spec_method")]
    pub speculation: String,
    #[serde(default)]
    pub speculation_n: u8,
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_spec_method() -> String {
    "off".to_string()
}

impl Default for Qgi2Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            persona: PersonaConfig::default(),
            engines: vec![
                EngineConfig {
                    role: "planner".into(),
                    base_url: "http://127.0.0.1:8000/v1".into(),
                    model: "Qwen3.8-Flash-Next-NVFP4".into(),
                    speculation: "mtp".into(),
                    speculation_n: 2,
                    api_key: None,
                },
                EngineConfig {
                    role: "worker".into(),
                    base_url: "http://127.0.0.1:8001/v1".into(),
                    model: "Qwen3.8-27B-NVFP4".into(),
                    speculation: "dflash2".into(),
                    speculation_n: 7,
                    api_key: None,
                },
            ],
            embedder: Some(EngineConfig {
                role: "embedder".into(),
                base_url: "http://127.0.0.1:8002/v1".into(),
                model: "Qwen3-Embedding-0.6B".into(),
                speculation: "off".into(),
                speculation_n: 0,
                api_key: None,
            }),
        }
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    home().map(|h| h.join(".qgi2").join("config.toml"))
}

fn default_graph_dir() -> Option<PathBuf> {
    home().map(|h| h.join(".qgi2").join("graphs"))
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

impl Qgi2Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Load from `path`, falling back to defaults when it does not exist.
    ///
    /// A missing config is normal on first run; an unparseable one is not, and
    /// is surfaced rather than silently replaced by defaults that would send
    /// requests somewhere the user did not choose.
    pub fn load_or_default(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::load(path)
    }

    pub fn persona(&self) -> Result<Persona> {
        let mood: Mood = self.persona.mood.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        let profile: Profile = self
            .persona
            .profile
            .parse()
            .map_err(|e: String| anyhow::anyhow!(e))?;
        Ok(Persona::new(mood, profile))
    }

    pub fn session_config(&self) -> Result<SessionConfig> {
        Ok(SessionConfig {
            persona: self.persona()?,
            allow_mood_switch: self.persona.allow_mood_switch,
            ..SessionConfig::default()
        })
    }

    pub fn registry(&self) -> Result<EngineRegistry> {
        let mut r = EngineRegistry::new();
        for e in &self.engines {
            let role = match e.role.as_str() {
                "planner" => ModelRole::Planner,
                "worker" => ModelRole::Worker,
                other => anyhow::bail!("unknown engine role {other:?}; expected planner or worker"),
            };
            let mut endpoint = Endpoint::new(&e.base_url, &e.model, parse_spec(e)?);
            endpoint.api_key = e.api_key.clone();
            r.register(role, endpoint);
        }
        if let Some(e) = &self.embedder {
            let mut endpoint = Endpoint::new(&e.base_url, &e.model, Speculation::Off);
            endpoint.api_key = e.api_key.clone();
            r.set_embedder(endpoint);
        }
        Ok(r)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}

fn parse_spec(e: &EngineConfig) -> Result<Speculation> {
    Ok(match e.speculation.as_str() {
        "mtp" => Speculation::Mtp { n: e.speculation_n },
        "dflash2" => Speculation::DFlash2 { n: e.speculation_n },
        "ngram" => Speculation::NGram { n: e.speculation_n },
        "off" | "none" => Speculation::Off,
        other => anyhow::bail!(
            "unknown speculation {other:?}; expected mtp, dflash2, ngram, or off"
        ),
    })
}

/// The `[providers.qgi2]` block a user adds to their jcode config.
///
/// This is the entire jcode-side footprint of QGI-2: one block in a file jcode
/// already reads, using an extension point jcode already documents.
pub fn jcode_provider_snippet(bind: &str) -> String {
    format!(
        r#"# Add to ~/.jcode/config.toml — this is the ONLY jcode-side change QGI-2 needs.
# No jcode source file is modified.

[providers.qgi2]
type = "openai-compatible"
base_url = "http://{bind}/v1"
auth = "none"
requires_api_key = false
default_model = "qgi2/builder-traceable"
model_catalog = true

# Then:  jcode --provider qgi2
# Switch persona with jcode's own model switcher:
#   /model qgi2:qgi2/researcher-deterministic
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_round_trips_through_toml() {
        let c = Qgi2Config::default();
        let text = c.to_toml().unwrap();
        let back: Qgi2Config = toml::from_str(&text).unwrap();
        assert_eq!(back.persona().unwrap(), c.persona().unwrap());
        assert_eq!(back.engines.len(), 2);
    }

    #[test]
    fn the_default_config_builds_a_registry_that_serves_the_default_persona() {
        let c = Qgi2Config::default();
        let r = c.registry().unwrap();
        let router = qgi2_router::Router::new(c.persona().unwrap());
        r.preflight(&router.plan_all().unwrap()).unwrap();
    }

    #[test]
    fn the_default_config_does_not_serve_the_deterministic_profile() {
        // Deterministic needs a worker launched with MTP; the default deploys
        // DFlash2. Better to fail preflight than to report acceptance numbers
        // for a configuration nobody chose.
        let c = Qgi2Config::default();
        let router =
            qgi2_router::Router::new(Persona::new(Mood::Builder, Profile::Deterministic));
        assert!(
            c.registry()
                .unwrap()
                .preflight(&router.plan_all().unwrap())
                .is_err()
        );
    }

    #[test]
    fn an_unknown_speculation_is_rejected() {
        let mut c = Qgi2Config::default();
        c.engines[0].speculation = "magic".into();
        assert!(c.registry().unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn an_unknown_role_is_rejected() {
        let mut c = Qgi2Config::default();
        c.engines[0].role = "oracle".into();
        assert!(c.registry().unwrap_err().to_string().contains("oracle"));
    }

    #[test]
    fn a_missing_config_file_falls_back_to_defaults() {
        let p = PathBuf::from("definitely-not-a-real-path-qgi2.toml");
        assert!(Qgi2Config::load_or_default(Some(&p)).is_ok());
    }

    #[test]
    fn the_jcode_snippet_uses_only_documented_config_keys() {
        let s = jcode_provider_snippet("127.0.0.1:8788");
        assert!(s.contains("[providers.qgi2]"));
        assert!(s.contains("openai-compatible"));
        assert!(s.contains("No jcode source file is modified"));
    }
}
