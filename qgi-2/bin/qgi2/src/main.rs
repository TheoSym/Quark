//! `qgi2` — the QGI-2 harness binary.
//!
//! Subcommands:
//!
//! - `serve`   run the OpenAI-compatible edge that stock jcode talks to
//! - `doctor`  check that every endpoint the persona needs is up and correctly
//!   configured, before a session discovers it on turn one
//! - `config`  print a default config, or the jcode snippet that wires it up
//! - `plan`    show the routing table for a persona: the explicit triple for
//!   every step, which is the spec's "nothing defaults" made visible

mod config;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::{Qgi2Config, default_config_path, jcode_provider_snippet};
use qgi2_edge_http::{AppState, SessionStore, router};
use qgi2_engine::{EngineKind, HiCacheConfig, HttpClient, engine_for, hicache};
use qgi2_router::Router as StepRouter;
use qgi2_spec_types::{Mood, Persona, Profile};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "qgi2", version, about = "Inference-first agent harness on jcode")]
struct Cli {
    /// Path to the QGI-2 config file (default: ~/.qgi2/config.toml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the OpenAI-compatible edge.
    Serve {
        /// Override the configured bind address.
        #[arg(long)]
        bind: Option<String>,
    },
    /// Check every endpoint the configured persona needs.
    Doctor {
        /// Check a specific persona instead of the configured one.
        #[arg(long)]
        mood: Option<String>,
        #[arg(long)]
        profile: Option<String>,
    },
    /// Print configuration.
    Config {
        /// Print the `[providers.qgi2]` block for ~/.jcode/config.toml.
        #[arg(long)]
        jcode: bool,
    },
    /// Show the routing table for a persona.
    Plan {
        #[arg(long, default_value = "builder")]
        mood: String,
        #[arg(long, default_value = "traceable")]
        profile: String,
    },
    /// SGLang HiCache: print launch commands, or read a live deployment's tiers.
    Hicache {
        /// Probe the configured endpoints and report per-tier hit rates.
        #[arg(long)]
        probe: bool,
        /// Which tier to configure for: 2 (host memory) or 3 (external store).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// L3 backend when --tier 3: `file` or `mooncake`.
        #[arg(long, default_value = "file")]
        l3: String,
        /// Mooncake master address, or the file backend's directory.
        #[arg(long)]
        store: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("QGI2_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let config_path = cli.config.clone().or_else(default_config_path);
    let cfg = Qgi2Config::load_or_default(config_path.as_deref())?;

    match cli.command {
        Command::Serve { bind } => serve(cfg, bind).await,
        Command::Doctor { mood, profile } => doctor(cfg, mood, profile).await,
        Command::Config { jcode } => {
            if jcode {
                print!("{}", jcode_provider_snippet(&cfg.server.bind));
            } else {
                print!("{}", cfg.to_toml()?);
            }
            Ok(())
        }
        Command::Plan { mood, profile } => plan(&cfg, &mood, &profile),
        Command::Hicache { probe, tier, l3, store } => {
            hicache_cmd(cfg, probe, tier, &l3, store).await
        }
    }
}

async fn serve(cfg: Qgi2Config, bind_override: Option<String>) -> Result<()> {
    let bind = bind_override.unwrap_or_else(|| cfg.server.bind.clone());
    let session_config = cfg.session_config()?;
    let registry = cfg.registry()?;

    // Fail before binding rather than on the first turn: a harness that accepts
    // connections and then errors on every request is worse than one that never
    // came up.
    let step_router = StepRouter::new(session_config.persona).with_speculation(
        session_config.planner_speculation,
        session_config.worker_speculation,
    );
    registry
        .preflight(&step_router.plan_all().map_err(|e| anyhow::anyhow!("{e}"))?)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("engine preflight failed; run `qgi2 doctor` for detail")?;

    let store = Arc::new(
        SessionStore::new(
            session_config,
            registry,
            cfg.skills()?,
            cfg.server.graph_dir.clone(),
        )
        .with_idle_timeout(std::time::Duration::from_secs(cfg.server.idle_minutes * 60)),
    );

    // Idle sweep: sessions that go quiet are ended -- promoted, decayed, their
    // durable slice merged -- rather than living until shutdown.
    {
        let store = store.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                tick.tick().await;
                match store.sweep().await {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(ended = n, "idle sessions ended"),
                    Err(e) => tracing::warn!(error = %e, "idle sweep failed"),
                }
            }
        });
    }
    let app = router(AppState {
        store: store.clone(),
    });

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("binding {bind}"))?;

    tracing::info!(%bind, "QGI-2 edge listening");
    tracing::info!("wire it up with: qgi2 config --jcode");

    let shutdown = {
        let store = store.clone();
        async move {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down; ending sessions and persisting graphs");
            // Session end promotes, decays, and writes metric facts. Skipping
            // it on shutdown would lose everything the session learned.
            match store.end_all().await {
                Ok(n) => tracing::info!(graphs = n, "persisted"),
                Err(e) => tracing::error!(error = %e, "failed to persist graphs"),
            }
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("serving")?;
    Ok(())
}

async fn doctor(cfg: Qgi2Config, mood: Option<String>, profile: Option<String>) -> Result<()> {
    let persona = match (mood, profile) {
        (None, None) => cfg.persona()?,
        (m, p) => {
            let mood: Mood = m
                .unwrap_or_else(|| cfg.persona.mood.clone())
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let profile: Profile = p
                .unwrap_or_else(|| cfg.persona.profile.clone())
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            Persona::new(mood, profile)
        }
    };

    println!("persona: {}/{}", persona.mood, persona.profile);

    let registry = cfg.registry()?;
    let session_config = cfg.session_config()?;
    // The same router the session will build, overrides included — a doctor
    // that routes differently from the harness is worse than no doctor.
    let step_router = StepRouter::new(persona).with_speculation(
        session_config.planner_speculation,
        session_config.worker_speculation,
    );
    let plans = step_router.plan_all().map_err(|e| anyhow::anyhow!("{e}"))?;

    for w in registry.unusual_pairings() {
        println!("note: {w}");
    }

    let mut failures = 0;
    println!("\nrouting:");
    for p in &plans {
        match registry.resolve(p.role, p.speculation) {
            Ok(e) => println!(
                "  {:<10} {:<8} {:<14} -> {} ({})",
                p.step.as_str(),
                p.role.as_str(),
                p.speculation.to_string(),
                e.base_url,
                e.engine
            ),
            Err(e) => {
                failures += 1;
                println!(
                    "  {:<10} {:<8} {:<14} -> UNROUTED",
                    p.step.as_str(),
                    p.role.as_str(),
                    p.speculation.to_string()
                );
                println!("      {e}");
                // Naming the flag turns "it doesn't work" into "run this".
                for kind in registry.engine_kinds() {
                    let engine = engine_for(kind, HttpClient::default());
                    if engine.supports(p.speculation) {
                        println!("      launch: {}", engine.launch_hint(p.speculation));
                    }
                }
            }
        }
    }

    println!("\nreachability:");
    let http = HttpClient::default();
    for (role, endpoint) in registry.all() {
        let detail = http.health_detail(endpoint).await;
        if detail.is_err() {
            failures += 1;
        }
        println!(
            "  {:<8} {:<8} {:<40} {}",
            role,
            endpoint.engine.as_str(),
            endpoint.base_url,
            match &detail {
                Ok(()) => "up".to_string(),
                Err(why) => format!("DOWN — {why}"),
            }
        );
    }
    if let Some(e) = &registry.embedder {
        let detail = http.health_detail(e).await;
        let up = detail.is_ok();
        println!(
            "  {:<8} {:<8} {:<40} {}",
            "embedder",
            e.engine.as_str(),
            e.base_url,
            match &detail {
                Ok(()) => "up".to_string(),
                Err(why) => format!("DOWN — {why}"),
            }
        );
        if !up && !persona.profile.retrieval().lexical_only {
            // Quick never calls the embedder, so its absence only matters for
            // the profiles that do.
            failures += 1;
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} problem(s); QGI-2 will not serve this persona");
    }
    println!("\nall checks passed");
    Ok(())
}

fn plan(cfg: &Qgi2Config, mood: &str, profile: &str) -> Result<()> {
    let persona = Persona::new(
        mood.parse().map_err(|e: String| anyhow::anyhow!(e))?,
        profile.parse().map_err(|e: String| anyhow::anyhow!(e))?,
    );
    // The same overrides the session will use. A routing table the harness
    // would not actually follow is worse than no table — it was showing MTP for
    // a planner the config had already pinned to something else.
    let session_config = cfg.session_config()?;
    let router = StepRouter::new(persona).with_speculation(
        session_config.planner_speculation,
        session_config.worker_speculation,
    );

    println!("persona: {}/{}", persona.mood, persona.profile);
    if cfg.registry()?.is_single_model() {
        println!(
            "mode:    single-model — one process serves both roles, so the \
             planner:worker token ratio is not checked"
        );
    }
    println!();
    println!(
        "{:<10} {:<8} {:<14} {:<7} {:<6} {:<8} schema",
        "step", "model", "speculation", "temp", "seed", "think"
    );
    for p in router.plan_all().map_err(|e| anyhow::anyhow!("{e}"))? {
        println!(
            "{:<10} {:<8} {:<14} {:<7.2} {:<6} {:<8} {}",
            p.step.as_str(),
            p.role.as_str(),
            p.speculation.to_string(),
            p.sampling.temperature,
            if p.sampling.seed.is_some() { "fixed" } else { "-" },
            if p.sampling.thinking { "on" } else { "off" },
            if p.schema.is_some() { "yes" } else { "free text" }
        );
    }
    Ok(())
}

/// `qgi2 hicache` — configure or inspect SGLang's hierarchical KV cache.
async fn hicache_cmd(
    cfg: Qgi2Config,
    probe: bool,
    tier: u8,
    l3: &str,
    store: Option<String>,
) -> Result<()> {
    if probe {
        return hicache_probe(cfg).await;
    }

    let preset = match tier {
        2 => HiCacheConfig::l2_only(),
        3 => match l3 {
            "file" => HiCacheConfig::with_file_l3(
                store.unwrap_or_else(|| "/var/cache/qgi2/kv".to_string()),
            ),
            "mooncake" => HiCacheConfig::with_mooncake_l3(
                store.ok_or_else(|| {
                    anyhow::anyhow!("--store is the mooncake master address, e.g. 10.0.0.1:50051")
                })?,
                hostname(),
            ),
            other => anyhow::bail!("unknown L3 backend {other:?}; expected file or mooncake"),
        },
        other => anyhow::bail!("unknown tier {other}; expected 2 (host memory) or 3 (store)"),
    };
    // A declared [hicache] block is the source of truth -- it carries page_size,
    // the GDN state pool and anything else the user sized. The tier preset only
    // fills in an L3 it does not name.
    let hc = match cfg.hicache.clone() {
        Some(mut declared) => {
            if declared.l3.is_none() {
                declared.l3 = preset.l3;
            }
            declared
        }
        None => preset,
    };

    let problems = hc.problems();
    if !problems.is_empty() {
        // A launch command generated from a configuration that will not behave
        // as written is worse than no command at all.
        for p in &problems {
            eprintln!("problem: {p}");
        }
        anyhow::bail!("{} problem(s) in the HiCache configuration", problems.len());
    }

    println!("# HiCache tiers: {}", 
        hc.tiers().iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" -> "));
    println!(
        "# The harness pads its stable prefix to {}-token pages when this is enabled.",
        hc.page_size
    );
    if let Some(g) = &hc.gdn {
        println!(
            "# GDN state pool: {} concurrent x {} slots ({} lazy + {} draft) x {:.1} MB = {:.1} GB",
            g.max_concurrent,
            g.slots_per_request(),
            g.lazy_slots,
            g.draft_slots(),
            g.slot_bytes as f64 / 1e6,
            g.pool_gb()
        );
        if let Some(kv) = g.kv_pool_gb() {
            println!("#   leaves {kv:.1} GB for the KV pool under the configured budget");
        }
    }
    println!("#\n# Add to ~/.qgi2/config.toml:\n#");
    for line in toml::to_string_pretty(&hc)?.lines() {
        println!("#   {line}");
    }
    println!("#   ^ under a [hicache] table\n");

    for e in &cfg.engines {
        if e.engine != "sglang" {
            continue;
        }
        let spec_flags = sglang_spec_flags(e);
        let port = port_of(&e.base_url).unwrap_or(30000);
        println!("# --- {} ({}) ---", e.role, e.base_url);
        println!("{}\n", hc.launch_command(&e.model, port, &spec_flags));
    }
    Ok(())
}

/// Read a live deployment's tier hit rates.
async fn hicache_probe(cfg: Qgi2Config) -> Result<()> {
    let registry = cfg.registry()?;
    let http = HttpClient::default();
    let declared = cfg.hicache.clone().unwrap_or_else(HiCacheConfig::l2_only);

    let mut any_sglang = false;
    for (role, endpoint) in registry.all() {
        if endpoint.engine != EngineKind::Sglang {
            continue;
        }
        any_sglang = true;
        let stats = hicache::scrape_hicache(&http, endpoint).await;
        println!("{role} @ {}", endpoint.base_url);
        if !stats.any() {
            // Distinguish "not enabled" from "unreachable": both show as no
            // data, and the fixes are different.
            println!("  no HiCache metrics — server may not have --enable-hierarchical-cache,");
            println!("  or /metrics is not exposed (SGLang needs --enable-metrics)");
            continue;
        }
        for (label, v) in [
            ("L1 (GPU)  ", stats.l1_hit_rate),
            ("L2 (host) ", stats.l2_hit_rate),
            ("L3 (store)", stats.l3_hit_rate),
        ] {
            match v {
                Some(r) => println!("  {label} hit rate {:>6.1}%", r * 100.0),
                None => println!("  {label} not reported"),
            }
        }
        if let Some(u) = stats.host_pool_utilization {
            println!("  host pool  {:>6.1}% full", u * 100.0);
        }
        for f in stats.findings(&declared) {
            println!("  ! {f}");
        }
    }

    if !any_sglang {
        println!("no SGLang endpoints configured; HiCache is SGLang-only");
    }
    Ok(())
}

fn sglang_spec_flags(e: &config::EngineConfig) -> Vec<String> {
    let n = e.speculation_n;
    match e.speculation.as_str() {
        "eagle3" => vec![
            "--speculative-algorithm".into(),
            "EAGLE3".into(),
            "--speculative-num-steps".into(),
            n.to_string(),
            "--speculative-eagle-topk".into(),
            "8".into(),
            "--speculative-num-draft-tokens".into(),
            (n + 1).to_string(),
        ],
        "mtp" => vec![
            "--speculative-algorithm".into(),
            "NEXTN".into(),
            "--speculative-num-steps".into(),
            n.to_string(),
            "--speculative-num-draft-tokens".into(),
            (n + 1).to_string(),
        ],
        "ngram" => vec![
            "--speculative-algorithm".into(),
            "NGRAM".into(),
            "--speculative-num-draft-tokens".into(),
            n.to_string(),
        ],
        _ => Vec::new(),
    }
}

fn port_of(base_url: &str) -> Option<u16> {
    base_url
        .rsplit(':')
        .next()?
        .split('/')
        .next()?
        .parse()
        .ok()
}

fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok()
}
