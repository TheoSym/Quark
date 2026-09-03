//! QGI-2's own config file, read from `~/.qgi2/config.toml` by default.
//!
//! Deliberately separate from `~/.jcode/config.toml`: QGI-2 does not modify
//! jcode, and that includes not appending its own keys to jcode's config
//! schema. The only jcode-side change a user makes is adding a
//! `[providers.qgi2]` block, which is jcode's existing, documented extension
//! point for OpenAI-compatible endpoints.

use anyhow::{Context, Result};
use qgi2_engine::{Endpoint, EngineKind, EngineRegistry, HiCacheConfig};
use qgi2_spec_types::{ModelRole, Mood, Persona, Profile, Speculation};
use qgi2_turn::SessionConfig;
use qgi2_turn::session::SkillCandidate;
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
    /// SGLang HiCache. Ignored by vLLM endpoints, which have their own prefix
    /// cache and no tiering.
    #[serde(default)]
    pub hicache: Option<HiCacheConfig>,
    /// Skills the rules may activate into segment 4.
    ///
    /// QGI-2's selection is graph-driven -- a skill activates when retrieval
    /// reaches a node it covers -- which is a different signal from jcode's own
    /// embedding-similarity activation. The two coexist: jcode keeps activating
    /// its skills its way, and this catalogue is what the harness renders into
    /// the volatile tail. Without it segment 4 is always empty, which was the
    /// case before this existed.
    #[serde(default)]
    pub skills: Vec<SkillConfig>,
}

/// One skill the rules may activate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillConfig {
    pub name: String,
    /// Node-name prefixes this skill covers, e.g. `file:` or `task:deploy`.
    #[serde(default)]
    pub subjects: Vec<String>,
    /// Moods this applies to. Empty means every mood.
    #[serde(default)]
    pub moods: Vec<String>,
    /// Skills that must also activate when this one does.
    #[serde(default)]
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind: String,
    /// Where fact graphs are persisted between runs.
    pub graph_dir: Option<PathBuf>,
    /// Minutes of inactivity after which a session is ended: promoted,
    /// decayed, its durable slice merged. A server has no natural "session
    /// end" otherwise, and those steps were only running at shutdown.
    #[serde(default = "default_idle_minutes")]
    pub idle_minutes: u64,
}

fn default_idle_minutes() -> u64 {
    30
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8788".to_string(),
            graph_dir: default_graph_dir(),
            idle_minutes: 30,
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
    /// `vllm` or `sglang`. Defaults to vLLM, which is what the spec names.
    #[serde(default = "default_engine_kind")]
    pub engine: String,
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
    /// Send an explicit `cache_control` breakpoint. Needed by Anthropic and
    /// Alibaba Qwen routes, which cache only where the request says to; without
    /// it they report `cached_tokens: 0` on a perfectly stable prefix and the
    /// harness raises a false alarm on the spec's key metric.
    #[serde(default)]
    pub cache_control: bool,
}

fn default_spec_method() -> String {
    "off".to_string()
}

fn default_engine_kind() -> String {
    "vllm".to_string()
}

impl Default for Qgi2Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            persona: PersonaConfig::default(),
            engines: vec![
                EngineConfig {
                    role: "planner".into(),
                    engine: "vllm".into(),
                    base_url: "http://127.0.0.1:8000/v1".into(),
                    model: "Qwen3.8-Flash-Next-NVFP4".into(),
                    speculation: "mtp".into(),
                    speculation_n: 2,
                    api_key: None,
                    cache_control: false,
                },
                EngineConfig {
                    role: "worker".into(),
                    engine: "vllm".into(),
                    base_url: "http://127.0.0.1:8001/v1".into(),
                    model: "Qwen3.8-27B-NVFP4".into(),
                    speculation: "dflash2".into(),
                    speculation_n: 7,
                    api_key: None,
                    cache_control: false,
                },
            ],
            hicache: None,
            skills: Vec::new(),
            embedder: Some(EngineConfig {
                role: "embedder".into(),
                engine: "vllm".into(),
                base_url: "http://127.0.0.1:8002/v1".into(),
                model: "Qwen3-Embedding-0.6B".into(),
                speculation: "off".into(),
                speculation_n: 0,
                api_key: None,
                cache_control: false,
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
        // The planner:worker ratio only means something when there are two
        // models. With one, it would describe the step mix and breach against a
        // target that no longer applies.
        let thresholds = if self.registry()?.is_single_model() {
            qgi2_spec_types::Thresholds::single_model()
        } else {
            qgi2_spec_types::Thresholds::default()
        };
        Ok(SessionConfig {
            persona: self.persona()?,
            thresholds,
            allow_mood_switch: self.persona.allow_mood_switch,
            page_alignment: self.hicache.as_ref().and_then(|h| h.page_alignment()),
            // Take the speculation each role's endpoint actually declares. The
            // spec's table assumes both models are self-hosted with a
            // speculator you control; a cloud-served planner has none, and
            // without this the router would plan MTP for it and nothing would
            // route. The endpoint is the ground truth either way.
            planner_speculation: self.declared_speculation("planner")?,
            worker_speculation: self.declared_speculation("worker")?,
            ..SessionConfig::default()
        })
    }

    /// The speculation declared for a role, when exactly one endpoint serves it.
    ///
    /// `None` when a role has several endpoints: that is the multi-process
    /// deployment the spec describes, and there the profile table is what picks
    /// between them.
    fn declared_speculation(&self, role: &str) -> Result<Option<Speculation>> {
        let mut matching = self.engines.iter().filter(|e| e.role == role);
        let Some(only) = matching.next() else {
            return Ok(None);
        };
        if matching.next().is_some() {
            return Ok(None);
        }
        Ok(Some(parse_spec(only)?))
    }

    pub fn registry(&self) -> Result<EngineRegistry> {
        let mut r = EngineRegistry::new();
        for e in &self.engines {
            let role = match e.role.as_str() {
                "planner" => ModelRole::Planner,
                "worker" => ModelRole::Worker,
                other => anyhow::bail!("unknown engine role {other:?}; expected planner or worker"),
            };
            let kind: EngineKind = e.engine.parse().map_err(|m: String| anyhow::anyhow!(m))?;
            let spec = parse_spec(e)?;
            // No capability check here. The endpoint's declared configuration is
            // ground truth: the QGI fleet serves DFlash2 on SGLang, which an
            // earlier hardcoded table called impossible, and a config error here
            // would have refused a live deployment. `qgi2 doctor` warns instead.
            let endpoint = Endpoint::new(&e.base_url, &e.model, spec)
                .with_engine(kind)
                .with_api_key(e.api_key.clone())
                .with_cache_control(e.cache_control);
            r.register(role, endpoint);
        }
        if let Some(e) = &self.embedder {
            let kind: EngineKind = e.engine.parse().map_err(|m: String| anyhow::anyhow!(m))?;
            r.set_embedder(
                Endpoint::new(&e.base_url, &e.model, Speculation::Off)
                    .with_engine(kind)
                    .with_api_key(e.api_key.clone()),
            );
        }
        Ok(r)
    }

    /// The skill catalogue as the rules consume it.
    pub fn skills(&self) -> Result<Vec<SkillCandidate>> {
        self.skills
            .iter()
            .map(|c| {
                let moods = c
                    .moods
                    .iter()
                    .map(|m| m.parse::<Mood>().map_err(|e| anyhow::anyhow!("skill {}: {e}", c.name)))
                    .collect::<Result<Vec<_>>>()?;
                let subjects: Vec<&str> = c.subjects.iter().map(String::as_str).collect();
                let requires: Vec<&str> = c.requires.iter().map(String::as_str).collect();
                Ok(SkillCandidate::new(&c.name)
                    .covering(&subjects)
                    .for_moods(&moods)
                    .requiring(&requires))
            })
            .collect()
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}

fn parse_spec(e: &EngineConfig) -> Result<Speculation> {
    Ok(match e.speculation.as_str() {
        "mtp" => Speculation::Mtp { n: e.speculation_n },
        "dflash2" => Speculation::DFlash2 { n: e.speculation_n },
        "eagle3" => Speculation::Eagle3 { n: e.speculation_n },
        "dspark" => Speculation::DSpark { n: e.speculation_n },
        "ngram" => Speculation::NGram { n: e.speculation_n },
        "off" | "none" => Speculation::Off,
        other => anyhow::bail!(
            "unknown speculation {other:?}; expected mtp, dflash2, eagle3, dspark, ngram, or off"
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

#[cfg(test)]
mod sglang_tests {
    use super::*;
    use qgi2_spec_types::{ModelRole, Speculation};

    fn sglang_config() -> Qgi2Config {
        Qgi2Config {
            engines: vec![
                EngineConfig {
                    role: "planner".into(),
                    engine: "sglang".into(),
                    base_url: "http://onyxtron-g12:30000/v1".into(),
                    model: "planner".into(),
                    speculation: "mtp".into(),
                    speculation_n: 2,
                    api_key: None,
                    cache_control: false,
                },
                EngineConfig {
                    role: "worker".into(),
                    engine: "sglang".into(),
                    base_url: "http://rhoditron-g24:30000/v1".into(),
                    model: "worker".into(),
                    speculation: "eagle3".into(),
                    speculation_n: 5,
                    api_key: None,
                    cache_control: false,
                },
            ],
            embedder: None,
            ..Qgi2Config::default()
        }
    }

    #[test]
    fn an_sglang_fleet_builds_a_registry() {
        let r = sglang_config().registry().unwrap();
        let e = r
            .resolve(ModelRole::Worker, Speculation::Eagle3 { n: 5 })
            .unwrap();
        assert_eq!(e.engine, EngineKind::Sglang);
    }

    #[test]
    fn an_sglang_endpoint_may_declare_dflash2() {
        // The QGI fleet serves Qwen3.8-27B with DFlash2 spec-decode on SGLang
        // (docs/MODELS.md, vidatron :18031). An earlier version rejected this
        // pairing at load from a hardcoded capability table, which would have
        // refused a live deployment. The declaration is ground truth; the
        // mismatch is surfaced as a warning by `qgi2 doctor` instead.
        let mut c = sglang_config();
        c.engines[1].speculation = "dflash2".into();
        c.engines[1].speculation_n = 7;
        let r = c.registry().expect("a declared pairing must load");
        assert_eq!(r.unusual_pairings().len(), 1, "but it is worth a warning");
    }

    #[test]
    fn a_mixed_fleet_is_allowed() {
        // An SGLang worker beside a vLLM planner is a reasonable deployment.
        let mut c = sglang_config();
        c.engines[0].engine = "vllm".into();
        let r = c.registry().unwrap();
        assert_eq!(r.engine_kinds().len(), 2);
    }

    #[test]
    fn an_unknown_engine_is_rejected() {
        let mut c = sglang_config();
        c.engines[0].engine = "tensorrt".into();
        assert!(c.registry().unwrap_err().to_string().contains("tensorrt"));
    }

    #[test]
    fn the_engine_field_defaults_to_vllm_for_older_configs() {
        let toml_text = r#"
[[engines]]
role = "planner"
base_url = "http://h:8000/v1"
model = "m"
speculation = "mtp"
speculation_n = 2
"#;
        let c: Qgi2Config = toml::from_str(toml_text).unwrap();
        assert_eq!(c.engines[0].engine, "vllm");
    }
}

#[cfg(test)]
mod shipped_configs {
    use super::*;

    #[test]
    fn the_selfhosted_config_sizes_the_gdn_pool() {
        // The shipped example must parse, and its [hicache.gdn] block must
        // reach the launch flags -- the wire spelling of `speculation` inside a
        // table is exactly what broke the first time this was written.
        let cfg: Qgi2Config =
            toml::from_str(include_str!("../../../config/qgi2.selfhosted.toml")).unwrap();
        let hc = cfg.hicache.expect("[hicache] declared");
        let g = hc.gdn.expect("[hicache.gdn] declared");
        assert_eq!(g.slots_per_request(), 12);
        let flags = hc.launch_flags().join(" ");
        assert!(flags.contains("--max-mamba-cache-size 96"), "{flags}");
        assert!(hc.problems().is_empty(), "{:?}", hc.problems());
    }
}
