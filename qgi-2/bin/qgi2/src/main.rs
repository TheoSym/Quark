//! `qgi2` — the QGI-2 harness binary.
//!
//! Subcommands:
//!
//! - `serve`   run the OpenAI-compatible edge that stock jcode talks to
//! - `doctor`  check that every endpoint the persona needs is up and correctly
//!             configured, before a session discovers it on turn one
//! - `config`  print a default config, or the jcode snippet that wires it up
//! - `plan`    show the routing table for a persona: the explicit triple for
//!             every step, which is the spec's "nothing defaults" made visible

mod config;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::{Qgi2Config, default_config_path, jcode_provider_snippet};
use qgi2_edge_http::{AppState, SessionStore, router};
use qgi2_engine_vllm::VllmClient;
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
        Command::Plan { mood, profile } => plan(&mood, &profile),
    }
}

async fn serve(cfg: Qgi2Config, bind_override: Option<String>) -> Result<()> {
    let bind = bind_override.unwrap_or_else(|| cfg.server.bind.clone());
    let session_config = cfg.session_config()?;
    let registry = cfg.registry()?;

    // Fail before binding rather than on the first turn: a harness that accepts
    // connections and then errors on every request is worse than one that never
    // came up.
    let step_router = StepRouter::new(session_config.persona);
    registry
        .preflight(&step_router.plan_all().map_err(|e| anyhow::anyhow!("{e}"))?)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("engine preflight failed; run `qgi2 doctor` for detail")?;

    let store = Arc::new(SessionStore::new(
        session_config,
        registry,
        Vec::new(),
        cfg.server.graph_dir.clone(),
    ));
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
    let step_router = StepRouter::new(persona);
    let plans = step_router.plan_all().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut failures = 0;
    println!("\nrouting:");
    for p in &plans {
        match registry.resolve(p.role, p.speculation) {
            Ok(e) => println!(
                "  {:<10} {:<8} {:<14} -> {}",
                p.step.as_str(),
                p.role.as_str(),
                p.speculation.to_string(),
                e.base_url
            ),
            Err(e) => {
                failures += 1;
                println!("  {:<10} {:<8} {:<14} -> UNROUTED", p.step.as_str(), p.role.as_str(), p.speculation.to_string());
                println!("      {e}");
            }
        }
    }

    println!("\nreachability:");
    let client = VllmClient::default();
    for (role, endpoint) in registry.all() {
        let up = client.health(endpoint).await;
        if !up {
            failures += 1;
        }
        println!(
            "  {:<8} {:<40} {}",
            role,
            endpoint.base_url,
            if up { "up" } else { "UNREACHABLE" }
        );
    }
    if let Some(e) = &registry.embedder {
        let up = client.health(e).await;
        println!(
            "  {:<8} {:<40} {}",
            "embedder",
            e.base_url,
            if up { "up" } else { "UNREACHABLE" }
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

fn plan(mood: &str, profile: &str) -> Result<()> {
    let persona = Persona::new(
        mood.parse().map_err(|e: String| anyhow::anyhow!(e))?,
        profile.parse().map_err(|e: String| anyhow::anyhow!(e))?,
    );
    let router = StepRouter::new(persona);

    println!("persona: {}/{}\n", persona.mood, persona.profile);
    println!(
        "{:<10} {:<8} {:<14} {:<7} {:<6} {:<8} {}",
        "step", "model", "speculation", "temp", "seed", "think", "schema"
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
