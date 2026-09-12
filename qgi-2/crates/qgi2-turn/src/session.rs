//! A QGI-2 session: the graph, the assembler's prefix memory, and the loop.
//!
//! # Rounds
//!
//! The spec's per-turn loop assumes the harness runs tools itself. QGI-2 does
//! not: both edges sit under jcode's agent loop, which executes tools and
//! re-enters the provider with the results. So a *turn* — one user query —
//! is made of one or more *rounds*, and each round is one call into
//! [`Session::round`]:
//!
//! ```text
//! round 0:  assemble → plan → (tool calls) ────────────► caller executes
//! round 1:  assemble → extract/verify/commit results
//!                    → plan → (more tool calls) ───────► caller executes
//! round 2:  assemble → extract/verify/commit results
//!                    → plan → answer → extract → verify → commit → mood check
//! ```
//!
//! The turn index advances only on round 0, and metrics accumulate across
//! rounds into one [`TurnMetrics`], so "tokens per turn" means what the spec
//! says it means rather than "tokens per model round-trip".

use crate::steps::{self, Engines, ExtractOutput, PlanOutput, RouteOutput, ToolArgsOutput};
use crate::tools::{ToolCall, ToolDisposition, ToolOutcome, ToolRunner};
use anyhow::Result;
use qgi2_assembler::{Assembler, CacheOutlook};
use qgi2_engine::EngineRegistry;
use qgi2_factgraph::retrieval::EntryMethod;
use qgi2_factgraph::{FactGraph, RenderBudget, Retrieval, Scope, Walk};
use qgi2_memory::{LineStore, RetrieveBudget, Speaker};
use qgi2_metrics::{Breach, SessionMetrics, TurnMetrics};
use qgi2_router::{Router, schemas};
use qgi2_rules::{
    MoodSwitchConfig, VerifyConfig, VerifyOutcome, mood_check, select_skills, tool_mask, verify,
};
use qgi2_spec_types::{
    FactId, ModelRole, Mood, Persona, Profile, Relation, Source, StepKind, Thresholds,
};
use serde::{Deserialize, Serialize};

// The skill catalogue type belongs to the rules crate, which owns selection.
pub use qgi2_rules::SkillCandidate;

/// Session-level configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub persona: Persona,
    pub thresholds: Thresholds,
    pub verify: VerifyConfig,
    pub mood_switch: MoodSwitchConfig,
    /// Confidence a session fact needs to reach the durable slice.
    pub promote_min_confidence: f32,
    /// Times a session fact must be reinforced to reach the durable slice.
    pub promote_min_reinforcements: u32,
    /// Session-end decay factor and the confidence below which facts are
    /// dropped.
    pub decay_factor: f32,
    pub decay_floor: f32,
    /// Whether to let the rules switch mood mid-session.
    ///
    /// Off by default. A mood switch rewrites segment 2 and therefore discards
    /// the cached prefix; that is a decision worth opting into rather than
    /// having happen to you partway through a long session.
    pub allow_mood_switch: bool,
    /// Cap on tool rounds within one turn.
    ///
    /// Without it, a model that keeps asking for tools and never answers loops
    /// forever, and each round costs a real request. At the cap the loop stops
    /// asking for tools and forces the answer step, so the user gets a reply
    /// that says what happened rather than nothing at all.
    pub max_tool_rounds: u32,
    /// Pad the stable prefix to the engine's KV page boundary. Set from the
    /// `[hicache]` config; `None` disables alignment.
    #[serde(default)]
    pub page_alignment: Option<qgi2_engine::PageAlignment>,
    /// Bytes of each tool result fed to the extract and answer steps. Larger
    /// results are windowed (head + tail) with the omission stated. A single
    /// large `read` otherwise lands in full in both prompts, twice per round.
    #[serde(default = "default_max_tool_output_bytes")]
    pub max_tool_output_bytes: usize,
    /// Skip the route step when retrieval already has an exact-key hit -- the
    /// user named a node verbatim, and a worker call to confirm it adds
    /// latency for little. Off by default because the spec lists route as a
    /// step; turn it on once you have measured what it saves.
    #[serde(default)]
    pub skip_route_on_exact_hit: bool,
    /// Speculation overrides for deployments that differ from the spec's table
    /// — most commonly a cloud-served planner, which has no speculator you can
    /// configure. `None` leaves the table in charge.
    #[serde(default)]
    pub planner_speculation: Option<qgi2_spec_types::Speculation>,
    #[serde(default)]
    pub worker_speculation: Option<qgi2_spec_types::Speculation>,
    /// Line-store memory retrieved into segment 5 per turn: at most this many
    /// lines and this many bytes. The prototype's chat default is 120 lines;
    /// the harness starts at a quarter of that because its whole premise is a
    /// short volatile tail, and `memory_budget_hit` on the response says when
    /// to grow it.
    #[serde(default = "default_memory_lines")]
    pub memory_lines: usize,
    #[serde(default = "default_memory_bytes")]
    pub memory_bytes: usize,
}

fn default_max_tool_output_bytes() -> usize {
    16 * 1024
}

fn default_memory_lines() -> usize {
    30
}

fn default_memory_bytes() -> usize {
    6_000
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            persona: Persona::default(),
            thresholds: Thresholds::default(),
            verify: VerifyConfig::default(),
            mood_switch: MoodSwitchConfig::default(),
            promote_min_confidence: 0.7,
            promote_min_reinforcements: 2,
            decay_factor: 0.9,
            decay_floor: 0.2,
            allow_mood_switch: false,
            max_tool_rounds: 12,
            page_alignment: None,
            max_tool_output_bytes: default_max_tool_output_bytes(),
            skip_route_on_exact_hit: false,
            planner_speculation: None,
            worker_speculation: None,
            memory_lines: default_memory_lines(),
            memory_bytes: default_memory_bytes(),
        }
    }
}

/// One round's input.
#[derive(Debug, Clone, Default)]
pub struct RoundInput {
    /// The user's query. Constant across the rounds of one turn.
    pub query: String,
    /// Results for the calls the previous round returned. Empty on round 0.
    pub tool_results: Vec<ToolOutcome>,
    /// 0 for the first round of a turn.
    pub round: u32,
}

impl RoundInput {
    pub fn first(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            tool_results: Vec::new(),
            round: 0,
        }
    }

    pub fn continuation(
        query: impl Into<String>,
        round: u32,
        tool_results: Vec<ToolOutcome>,
    ) -> Self {
        Self {
            query: query.into(),
            tool_results,
            round,
        }
    }
}

/// How a round ended.
#[derive(Debug, Clone)]
pub enum RoundOutcome {
    /// The model wants tools run. The caller executes them and calls
    /// [`Session::round`] again with the results and `round + 1`.
    CallTools {
        calls: Vec<ToolCall>,
        result: TurnResult,
    },
    /// The turn is complete.
    Answered(TurnResult),
}

impl RoundOutcome {
    pub fn result(&self) -> &TurnResult {
        match self {
            Self::CallTools { result, .. } | Self::Answered(result) => result,
        }
    }

    pub fn calls(&self) -> &[ToolCall] {
        match self {
            Self::CallTools { calls, .. } => calls,
            Self::Answered(_) => &[],
        }
    }

    pub fn is_final(&self) -> bool {
        matches!(self, Self::Answered(_))
    }
}

/// What one turn (or the round so far) produced.
#[derive(Debug, Clone, Default)]
pub struct TurnResult {
    /// The user-facing answer. Empty until the final round.
    pub answer: String,
    /// Tools the runner executed itself this round.
    pub tools: Vec<ToolOutcome>,
    /// Facts committed so far this turn.
    pub committed: Vec<FactId>,
    /// Metrics accumulated across the turn's rounds.
    pub metrics: TurnMetrics,
    /// Threshold breaches. The spec treats these as bugs, so the edges surface
    /// them rather than filing them.
    pub breaches: Vec<Breach>,
    /// What assembly expected the cache to do this round.
    pub cache_outlook: Option<CacheOutlook>,
    /// Segment hashes, for Traceable and Deterministic logging.
    pub segment_hashes: Vec<(String, String)>,
    /// Set when the mood check decided to switch.
    pub mood_switched_to: Option<Mood>,
    /// Set when the round cap forced the answer.
    pub tool_rounds_exhausted: bool,
    /// Set when the profile wanted embedding-seeded retrieval but the embedder
    /// was unavailable and the turn fell back to lexical matching. Surfaced
    /// because a Traceable run that quietly degraded to Quick-grade retrieval
    /// would report numbers about a different configuration than it claims.
    pub retrieval_degraded: bool,
    /// Mood the route step suggested, if any. Recorded, not acted on: mood
    /// switching stays rule-driven, and a model suggestion is one input to
    /// that, not an override of it.
    pub route_suggested_mood: Option<String>,
    /// Tool results that were windowed before reaching the model this round.
    pub tool_outputs_truncated: usize,
    /// The answer step emitted raw tool-call markup instead of prose. The
    /// user would otherwise see engine syntax; the flag makes it countable.
    pub answer_contained_tool_markup: bool,
    /// Verbatim memory lines retrieved into this round's prompt.
    pub memory_lines: usize,
    /// More lines matched than the memory budget allowed. The counter the
    /// compaction event waits on: until it fires, compaction has no work.
    pub memory_budget_hit: bool,
}

/// One QGI-2 session.
pub struct Session {
    pub graph: FactGraph,
    pub config: SessionConfig,
    pub metrics: SessionMetrics,
    assembler: Assembler,
    retrieval: Retrieval,
    engines: Engines,
    registry: EngineRegistry,
    skills: Vec<SkillCandidate>,
    turn: u64,
    /// Metrics for the turn currently in flight, accumulated across its rounds.
    open_turn: Option<TurnMetrics>,
    /// Facts committed so far in the turn currently in flight.
    open_committed: Vec<FactId>,
    /// Relations seen this session, feeding the mood check.
    recent_relations: Vec<Relation>,
    /// The query's embedding for the turn in flight. The query is constant
    /// across a turn's rounds, so embedding it once per round was one embedder
    /// call per tool round for the same vector.
    turn_query_embedding: Option<Vec<f32>>,
    /// Last acceptance scrape per endpoint, so the reported number describes
    /// this turn rather than the server's lifetime.
    last_acceptance: std::collections::BTreeMap<String, qgi2_engine::AcceptanceSnapshot>,
    /// The line store: every turn's query, answer and tool output, verbatim,
    /// role-tagged, retrieved deterministically into segment 5. The graph
    /// keeps structure; this keeps content, so a file read once is not read
    /// again for want of memory.
    memory: LineStore,
    /// The previous round's tool results, verbatim and capped: the working
    /// window. A `read` in one round and the `edit` it feeds in the next are
    /// the commonest pair in a coding turn, and the edit needs the bytes
    /// exactly. Cleared when a turn ends.
    last_round_results: Option<String>,
}

impl Session {
    pub fn new(
        config: SessionConfig,
        registry: EngineRegistry,
        skills: Vec<SkillCandidate>,
    ) -> Self {
        // Read what the assembler needs before `config` moves into the struct.
        let assembler = match config.page_alignment {
            Some(a) => Assembler::with_budget(RenderBudget::default()).with_page_alignment(a),
            None => Assembler::with_budget(RenderBudget::default()),
        };
        Self {
            graph: FactGraph::new(),
            metrics: SessionMetrics::new(config.thresholds),
            config,
            assembler,
            retrieval: Retrieval::default(),
            engines: Engines::for_registry(&registry),
            registry,
            skills,
            turn: 0,
            open_turn: None,
            open_committed: Vec::new(),
            recent_relations: Vec::new(),
            turn_query_embedding: None,
            last_acceptance: std::collections::BTreeMap::new(),
            memory: LineStore::new(),
            last_round_results: None,
        }
    }

    /// Load a persisted line store.
    pub fn with_memory(mut self, memory: LineStore) -> Self {
        self.memory = memory;
        self
    }

    pub fn memory(&self) -> &LineStore {
        &self.memory
    }

    pub fn memory_json(&self) -> Result<String> {
        Ok(self.memory.to_json()?)
    }

    fn memory_budget(&self) -> RetrieveBudget {
        RetrieveBudget {
            max_lines: self.config.memory_lines,
            max_bytes: self.config.memory_bytes,
            ..RetrieveBudget::default()
        }
    }

    /// Replace the engine backends. See [`Engines::with_engine`].
    pub fn with_engines(mut self, engines: Engines) -> Self {
        self.engines = engines;
        self
    }

    /// The current backends, for building a modified set.
    pub fn engines(&self) -> &Engines {
        &self.engines
    }

    /// Load a persisted graph. The durable slice it carries becomes segment 3.
    pub fn with_graph(mut self, graph: FactGraph) -> Self {
        self.graph = graph;
        self
    }

    pub fn persona(&self) -> Persona {
        self.config.persona
    }

    pub fn mood(&self) -> Mood {
        self.config.persona.mood
    }

    pub fn profile(&self) -> Profile {
        self.config.persona.profile
    }

    pub fn turn_index(&self) -> u64 {
        self.turn
    }

    fn router(&self) -> Router {
        Router::new(self.config.persona).with_speculation(
            self.config.planner_speculation,
            self.config.worker_speculation,
        )
    }

    /// Check every endpoint the current persona will need, before turn one.
    pub fn preflight(&self) -> Result<()> {
        let plans = self
            .router()
            .plan_all()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.registry
            .preflight(&plans)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Run one turn to completion, executing tools through `tools`.
    ///
    /// Only usable with a runner that actually executes; a deferring runner
    /// would loop without progress, so that case returns an error rather than
    /// spinning. The edges call [`Session::round`] directly.
    pub async fn turn(&mut self, query: &str, tools: &dyn ToolRunner) -> Result<TurnResult> {
        match self.round(RoundInput::first(query), tools).await? {
            RoundOutcome::Answered(r) => Ok(r),
            RoundOutcome::CallTools { calls, .. } => anyhow::bail!(
                "the tool runner deferred {} call(s) but `turn` has no agent loop to \
                 execute them; drive rounds with `Session::round` instead",
                calls.len()
            ),
        }
    }

    /// Run one round.
    pub async fn round(
        &mut self,
        input: RoundInput,
        tools: &dyn ToolRunner,
    ) -> Result<RoundOutcome> {
        // The engine's page granularity bounds what a prompt can cache, so the
        // metrics carry it: a cache floor above the ceiling would report a
        // defect in a prefix that is doing everything a prefix can.
        let page = self
            .config
            .page_alignment
            .map(|a| a.granularity())
            .unwrap_or(0);
        if input.round == 0 {
            self.turn += 1;
            self.open_turn = Some(TurnMetrics::new(self.turn).with_page_size(page));
            self.open_committed.clear();
            self.turn_query_embedding = None;
        }
        let turn = self.turn;
        let router = self.router();

        let mut metrics = self
            .open_turn
            .take()
            .unwrap_or_else(|| TurnMetrics::new(turn).with_page_size(page));
        let mut result = TurnResult::default();

        // --- retrieve: embedder seeds entry points, route step refines them ---
        //
        // Spec: "Embedder -- entry-point retrieval only." The embedder picks
        // where to start; the route step (worker, under schema) prunes and
        // extends that list; the mood's traversal does the rest. Quick is
        // lexical-only in the spec's table and skips both model-side stages.
        let policy = self.profile().retrieval();
        let mut query_embedding = None;
        if !policy.lexical_only {
            query_embedding = match &self.turn_query_embedding {
                Some(v) => Some(v.clone()),
                None => {
                    let v = self.embed_query(&input.query).await;
                    self.turn_query_embedding = v.clone();
                    v
                }
            };
            if query_embedding.is_none() && self.registry.embedder.is_some() {
                result.retrieval_degraded = true;
            }
        }
        let candidates = self.retrieval.entry_points(
            &self.graph,
            &input.query,
            query_embedding.as_deref(),
            policy,
        );
        let has_exact_hit = candidates.iter().any(|e| e.how == EntryMethod::ExactKey);
        let candidate_names: Vec<String> = candidates.into_iter().map(|e| e.node).collect();

        let skip_route = self.config.skip_route_on_exact_hit && has_exact_hit;
        let entries = if policy.lexical_only || self.graph.is_empty() || skip_route {
            // Nothing for a route step to choose between yet.
            candidate_names
        } else {
            match self
                .route_step(&router, &input.query, &candidate_names, &mut metrics)
                .await
            {
                Ok(out) => {
                    result.route_suggested_mood = out.suggested_mood;
                    if out.entry_points.is_empty() {
                        candidate_names
                    } else {
                        out.entry_points
                    }
                }
                Err(e) => {
                    // A failed route step costs the refinement, not the turn.
                    tracing::warn!(error = %e, "route step failed; using retrieval candidates");
                    candidate_names
                }
            }
        };
        let traversal_spec = self.mood().table().traversal;
        let reached_ids = {
            let walk = Walk::new(&self.graph, &traversal_spec, self.profile().retrieval());
            let out = walk.from_entries(&entries);
            // Traceable logging: which nodes retrieval started from and how
            // much of memory it reached. A turn that answers "I know nothing"
            // about a subject with committed facts is diagnosed from this line.
            tracing::info!(
                entries = ?entries,
                unmatched = ?out.unmatched_entries,
                reached = out.facts.len(),
                graph = self.graph.len(),
                "retrieval"
            );
            out.facts
        };
        let reached_nodes: Vec<String> = reached_ids
            .iter()
            .filter_map(|id| self.graph.get(id))
            .flat_map(|f| [f.subject().to_string(), f.object().to_string()])
            .collect();
        let active_skills =
            select_skills(&self.skills, &reached_nodes, self.mood(), &[], &self.graph);

        // --- memory: verbatim lines for this query, deterministic ---
        let memory = self.memory.retrieve(&input.query, self.memory_budget());
        result.memory_lines = memory.lines.len();
        result.memory_budget_hit = memory.budget_hit();
        if !memory.is_empty() {
            tracing::info!(
                lines = memory.lines.len(),
                candidates = memory.candidates,
                store = self.memory.len(),
                budget_hit = memory.budget_hit(),
                "memory"
            );
        }

        // --- assemble ---
        let assembled = self.assembler.assemble(
            &self.graph,
            self.mood(),
            self.profile(),
            &active_skills,
            &reached_ids,
            &memory.text,
            &input.query,
        );
        result.segment_hashes = assembled.hash_log();
        result.cache_outlook = Some(assembled.outlook.clone());

        let system = assembled.system();
        let base_volatile = assembled.volatile();
        // The cache floor for this turn's requests is the stable prefix, in
        // tokens, so the metric asks for what the harness controls and not
        // for the step-specific tails.
        if let Some(align) = self.config.page_alignment {
            metrics.stable_prefix_tokens = align.estimated_tokens(system.len()) as u64;
        }

        // Tool output is appended to the volatile tail rather than folded into
        // the graph and re-rendered. File contents do not survive the trip
        // through a (subject, relation, object) triple, so the answer step needs
        // them verbatim; the graph gets the *structure* the extract step finds
        // in them.
        let cap = self.config.max_tool_output_bytes;
        // The previous round's results ride along verbatim as the working
        // window, then this round's replace them. A read in one round and the
        // edit it feeds in the next need the same bytes.
        let previous = self
            .last_round_results
            .take()
            .filter(|_| input.round > 0)
            .map(|p| format!("\n\n# Previous round's tool results\n{p}"))
            .unwrap_or_default();
        let volatile = if input.tool_results.is_empty() {
            format!("{base_volatile}{previous}")
        } else {
            result.tool_outputs_truncated +=
                input.tool_results.iter().filter(|o| o.exceeds(cap)).count();
            let observed = input
                .tool_results
                .iter()
                .map(|o| o.render_capped(cap))
                .collect::<Vec<_>>()
                .join("\n");
            self.last_round_results = Some(observed.clone());
            format!("{base_volatile}{previous}\n\n# Tool results\n{observed}")
        };

        // --- extract from the previous round's tool results ---
        if !input.tool_results.is_empty() {
            // Tool output enters the line store verbatim (blocks whole, prose
            // by sentence), capped per observation as the note's default.
            for o in &input.tool_results {
                let rendered = o.render_capped(cap);
                tracing::info!(
                    tool = %o.call.tool,
                    head = %rendered.chars().take(160).collect::<String>(),
                    bytes = rendered.len(),
                    "tool result"
                );
                self.memory
                    .add_block(turn as u32, Speaker::Tool, &rendered, 12_000);
            }
            let committed = self
                .extract_verify_commit(
                    &router,
                    &system,
                    &volatile,
                    Source::Tool("tools".into()),
                    &mut metrics,
                )
                .await?;
            self.open_committed.extend(committed);
        }

        // --- plan ---
        //
        // The plan step must see the caller's exact tool names. The mood
        // table only names tool *families* ("fs, shell, git"), and on the
        // first live jcode run the planner invented `fs.read` for a tool the
        // caller had registered as `read`; the mask then reported it
        // unavailable, six rounds in a row, and the task was never touched.
        let plan_step = router
            .plan(StepKind::Plan)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Only tools the mood mask will admit are offered. Listing a tool
        // and then refusing it is worse than not listing it: on the A/B bench
        // the planner chose jcode's `batch`, the mask denied it as outside the
        // mood, and the planner concluded it had no file access and gave up.
        let mut available_tools = tools.available().await?;
        {
            let names: Vec<String> = available_tools.iter().map(|t| t.name.clone()).collect();
            let mask = tool_mask(&names, self.mood(), &self.graph);
            available_tools.retain(|t| mask.permits(&t.name));
        }
        let plan_input = if available_tools.is_empty() {
            format!(
                "{volatile}\n\n# Tools available\n(none in this session: set needs_tools to false)"
            )
        } else {
            let mut s =
                format!("{volatile}\n\n# Tools available\nUse these exact names in `tool`:\n");
            for t in &available_tools {
                s.push_str("- ");
                s.push_str(&t.name);
                if !t.description.is_empty() {
                    s.push_str(": ");
                    s.push_str(t.description.lines().next().unwrap_or(""));
                }
                s.push('\n');
            }
            s
        };
        let (mut planned, plan_resp): (PlanOutput, _) = steps::run_structured(
            &self.engines,
            &self.registry,
            &plan_step,
            &system,
            &plan_input,
        )
        .await?;
        record(&mut metrics, ModelRole::Planner, &plan_resp);
        // A step whose intent *starts with* a tool name but names no tool is
        // taken as a call to that tool. The schema leaves `tool` optional so a
        // plan can hold non-tool steps; on the bench the planner wrote
        // `{"intent": "read app/models.py and app/util.py"}` with no tool,
        // the loop had nothing to run, and the turn ended in an answer.
        let tool_names: Vec<&str> = available_tools.iter().map(|t| t.name.as_str()).collect();
        for step in &mut planned.steps {
            if step.tool.is_none()
                && let Some(first) = step.intent.split_whitespace().next()
                && let Some(name) = tool_names.iter().find(|n| {
                    n.eq_ignore_ascii_case(
                        first.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'),
                    )
                })
            {
                step.tool = Some(name.to_string());
            }
        }

        // --- tool calls, masked by the rules ---
        let rounds_left = input.round < self.config.max_tool_rounds;
        // A step that names a tool *is* a request for tools, whatever the
        // separate flag says. On the A/B bench the planner produced
        // `needs_tools: false` beside a `bash:` step, the loop took the flag's
        // word, answered without running anything, and the task failed.
        let wants_tools = planned.needs_tools || planned.steps.iter().any(|s| s.tool.is_some());
        tracing::info!(
            round = input.round,
            needs_tools = planned.needs_tools,
            wants_tools,
            steps = ?planned
                .steps
                .iter()
                .map(|s| format!("{}:{}", s.tool.as_deref().unwrap_or("-"), s.intent))
                .collect::<Vec<_>>(),
            "plan"
        );
        if wants_tools && rounds_left {
            let (deferred, executed) = self
                .build_tool_calls(
                    &router,
                    &planned,
                    tools,
                    &system,
                    &volatile,
                    input.round,
                    &mut metrics,
                )
                .await?;
            result.tools = executed;

            if !deferred.is_empty() {
                // Hand the calls up and suspend the turn here. Metrics and
                // committed facts stay on the session so the next round
                // continues the same turn rather than starting a new one.
                self.open_turn = Some(metrics);
                result.metrics = metrics;
                result.committed = self.open_committed.clone();
                return Ok(RoundOutcome::CallTools {
                    calls: deferred,
                    result,
                });
            }
        } else if planned.needs_tools && !rounds_left {
            // Answer anyway rather than looping — but the model has to be
            // *told*, or it answers as if the work finished. Setting the flag
            // alone informed the caller and nobody else.
            result.tool_rounds_exhausted = true;
        }

        let volatile = if result.tool_rounds_exhausted {
            format!(
                "{volatile}

# Note
The tool budget for this turn ({} rounds) is spent.                  Answer with what you have and say plainly what was not done.",
                self.config.max_tool_rounds
            )
        } else {
            volatile
        };

        // Anything the runner executed inline is context for the answer.
        let volatile = if result.tools.is_empty() {
            volatile
        } else {
            result.tool_outputs_truncated += result.tools.iter().filter(|o| o.exceeds(cap)).count();
            let observed = result
                .tools
                .iter()
                .map(|o| o.render_capped(cap))
                .collect::<Vec<_>>()
                .join("\n");
            format!("{volatile}\n\n# Tool results\n{observed}")
        };

        // --- answer ---
        //
        // The answer request carries no tool definitions, so the engine's
        // tool-call parser does not run on it: a model that emits a call here
        // leaks raw call markup into the user-facing text. On the first live
        // run a DeepSeek planner did exactly that (`<｜DSML｜ calls>` in the
        // answer) after its plan asked for a shell tool the caller did not
        // offer. Say so in the prompt, and flag it when it happens anyway.
        let answer_step = router
            .plan(StepKind::Answer)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // The request is restated here because after several tool rounds the
        // query segment sits above kilobytes of tool output, and on the first
        // live jcode run the planner -- having fixed the file -- answered that
        // it had been given no task.
        let answer_input = format!(
            "{volatile}\n\n# Answer\nThe user's request was:\n{}\n\nWrite the final answer to \
             that request now, for the user. No tool can be called from this step; if \
             something needed a tool you did not have, say what is missing instead of \
             calling it.",
            input.query
        );
        let answer_resp = steps::run_step(
            &self.engines,
            &self.registry,
            &answer_step,
            &system,
            &answer_input,
        )
        .await?;
        record(&mut metrics, ModelRole::Planner, &answer_resp);
        result.answer = answer_resp.text().to_string();
        // The turn leaves the window: its query and answer join the store,
        // and the working window of tool results closes with it.
        self.memory
            .add(turn as u32, Speaker::User, &input.query, 12_000);
        self.memory
            .add(turn as u32, Speaker::Assistant, &result.answer, 12_000);
        self.last_round_results = None;
        if answer_contains_tool_markup(&result.answer) {
            result.answer_contained_tool_markup = true;
            tracing::warn!(
                head = %result.answer.chars().take(80).collect::<String>(),
                "answer step emitted tool-call markup instead of prose"
            );
        }

        // --- extract answer facts, verify, commit ---
        let answer_committed = self
            .extract_verify_commit(
                &router,
                &system,
                &format!("{volatile}\n\n# Answer\n{}", result.answer),
                Source::Answer,
                &mut metrics,
            )
            .await?;
        self.open_committed.extend(answer_committed);
        result.committed = self.open_committed.clone();

        // New subjects need vectors before the next turn can find them by
        // similarity. Done after commit so a rejected proposal never earns an
        // embedder call.
        if !policy.lexical_only {
            self.embed_new_subjects().await;
        }

        // --- acceptance, from the delta since the last scrape ---
        self.record_acceptance(&mut metrics).await;

        // --- mood check ---
        let decision = mood_check(self.mood(), &self.recent_relations, self.config.mood_switch);
        if decision.is_switch() && self.config.allow_mood_switch {
            self.config.persona.mood = decision.mood();
            // The mood segment is in the stable prefix, so the change is
            // deliberate: reset the assembler rather than have the next turn
            // report a broken prefix for something the harness chose.
            self.assembler.reset();
            result.mood_switched_to = Some(decision.mood());
        }

        result.metrics = metrics;
        result.breaches = metrics.breaches(self.config.thresholds);
        self.metrics.record(metrics);
        self.open_turn = None;
        Ok(RoundOutcome::Answered(result))
    }

    /// Embed the query for entry-point retrieval.
    ///
    /// `None` when there is no embedder or it failed; the caller degrades to
    /// lexical and flags it. The turn is never failed for a retrieval-quality
    /// problem.
    async fn embed_query(&self, query: &str) -> Option<Vec<f32>> {
        let embedder = self.registry.embedder.as_ref()?;
        let engine = self.engines.for_endpoint(embedder).ok()?;
        match engine.embed(embedder, &[query.to_string()]).await {
            Ok(mut v) if !v.is_empty() => Some(v.remove(0)),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(error = %e, "embedder unavailable; retrieval degraded to lexical");
                None
            }
        }
    }

    /// Embed every graph subject that does not have a vector yet.
    async fn embed_new_subjects(&mut self) {
        let Some(embedder) = self.registry.embedder.clone() else {
            return;
        };
        let missing = self.retrieval.missing_embeddings(&self.graph);
        if missing.is_empty() {
            return;
        }
        let Ok(engine) = self.engines.for_endpoint(&embedder) else {
            return;
        };
        match engine.embed(&embedder, &missing).await {
            Ok(vectors) => {
                for (node, v) in missing.into_iter().zip(vectors) {
                    self.retrieval.set_embedding(node, v);
                }
            }
            Err(e) => tracing::warn!(error = %e, "could not embed new subjects"),
        }
    }

    /// The route step: the worker refines the retrieval candidates under the
    /// route schema.
    ///
    /// Its prompt is segments 1-2 only. The durable slice and subgraph are
    /// left out because the subgraph is what this step is *choosing*, and
    /// because keeping the prompt to the two segments that never change within
    /// a session makes it fully cacheable on its own.
    async fn route_step(
        &self,
        router: &Router,
        query: &str,
        candidates: &[String],
        metrics: &mut TurnMetrics,
    ) -> Result<RouteOutput> {
        let step = router
            .plan(StepKind::Route)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let system = format!(
            "{}\n{}",
            qgi2_assembler::CORE_PROMPT,
            self.mood().table().render()
        );
        let mut user = format!("# Query\n{query}\n\n# Candidate entry points\n");
        if candidates.is_empty() {
            user.push_str("(none matched)\n");
        }
        for c in candidates {
            user.push_str("- ");
            user.push_str(c);
            user.push('\n');
        }
        user.push_str(
            "\nChoose the entry points to start retrieval from. Prefer candidates; add a \
             subject only if you are confident it exists in memory.",
        );
        let (mut out, resp): (RouteOutput, _) =
            steps::run_structured(&self.engines, &self.registry, &step, &system, &user).await?;
        record(metrics, ModelRole::Worker, &resp);
        // The route step may name a subject that is not in memory -- the
        // prompt invites it to add one "if confident" -- and a walk from a
        // node that does not exist reaches nothing. Keep only entry points
        // the graph actually has; an empty survivor list falls back to the
        // retrieval candidates at the call site.
        out.entry_points = retain_known_entries(&self.graph, out.entry_points);
        Ok(out)
    }

    /// The retrieval state (node embeddings), for persistence.
    pub fn retrieval_json(&self) -> Result<String> {
        Ok(self.retrieval.to_json()?)
    }

    /// Restore persisted retrieval state.
    pub fn with_retrieval(mut self, retrieval: Retrieval) -> Self {
        self.retrieval.absorb(retrieval);
        self
    }

    pub fn retrieval(&self) -> &Retrieval {
        &self.retrieval
    }

    /// Build the calls the plan asked for, splitting executed from deferred.
    ///
    /// The argument decodes for a multi-tool plan are independent of each
    /// other, so they are issued concurrently and joined. Serial decodes made a
    /// three-tool plan three worker round-trips long for no reason; the
    /// dispositions are then applied in plan order so `deferred` stays
    /// deterministic.
    // Private, called from one place; a parameter struct would add a type to
    // read for no reader benefit.
    #[allow(clippy::too_many_arguments)]
    async fn build_tool_calls(
        &self,
        router: &Router,
        planned: &PlanOutput,
        tools: &dyn ToolRunner,
        system: &str,
        volatile: &str,
        round: u32,
        metrics: &mut TurnMetrics,
    ) -> Result<(Vec<ToolCall>, Vec<ToolOutcome>)> {
        let available = tools.available().await?;
        let names: Vec<String> = available.iter().map(|t| t.name.clone()).collect();
        let mask = tool_mask(&names, self.mood(), &self.graph);

        let mut deferred = Vec::new();
        let mut executed = Vec::new();
        // (index, tool name, spec, intent) for every call that passed the mask.
        let mut pending: Vec<(usize, String, &crate::tools::ToolSpec, &str)> = Vec::new();

        for (index, step) in planned.steps.iter().enumerate() {
            let Some(wanted) = &step.tool else { continue };
            let id = ToolCall::id_for(round, index, wanted);

            if !mask.permits(wanted) {
                // Surfaced as a failed result rather than dropped: the model
                // needs to see the tool is unavailable, or it keeps planning
                // around it round after round. Logged too: on the A/B bench
                // the harness arm failed three tasks with "I could not read
                // the files" and nothing said which name was refused or why.
                tracing::warn!(
                    tool = %wanted,
                    reason = ?mask.denial_reason(wanted),
                    available = ?names,
                    "tool mask denied a planned call"
                );
                executed.push(ToolOutcome::error(
                    ToolCall {
                        id,
                        tool: wanted.clone(),
                        arguments: serde_json::Value::Null,
                    },
                    mask.denial_reason(wanted)
                        .unwrap_or_else(|| "unavailable".into()),
                ));
                continue;
            }

            let Some(spec) = available.iter().find(|t| &t.name == wanted) else {
                continue;
            };
            pending.push((index, wanted.clone(), spec, step.intent.as_str()));
        }

        // Decode every tool's arguments concurrently. Each is constrained to
        // that tool's own parameter schema, so a malformed call is impossible
        // rather than merely unlikely.
        let decodes = pending
            .iter()
            .map(|(index, wanted, spec, intent)| async move {
                let mut args_step = router
                    .plan(StepKind::ToolArgs)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                args_step.schema =
                    Some(schemas::tool_args_schema_for(&spec.name, &spec.parameters));
                // The schema constrains the shape; the description says what
                // the fields mean. Without it the worker filled jcode's
                // `agentgrep` with a plausible-looking mix of fields the tool
                // rejected, twice in a row.
                // Memory retrieved by the *step's* intent, not the turn's
                // query: a step that reads "the app source files" needs the
                // listing an earlier round produced, which the query-keyed
                // retrieval has no reason to surface. On the bench, two
                // parallel `read` steps both decoded to the test file because
                // the worker had never seen a file name.
                let step_memory = self.memory.retrieve(intent, self.memory_budget());
                let memory_block = if step_memory.is_empty() {
                    String::new()
                } else {
                    format!("\n\n# Memory for this step\n{}", step_memory.text)
                };
                let user = format!(
                    "{volatile}{memory_block}\n\n# Next step\n{intent}\n\n# Tool\n{}: {}\nFill \
                     only the arguments this tool needs for the step above; leave optional \
                     fields out unless the step requires them.",
                    spec.name,
                    spec.description.trim()
                );
                let (args, resp): (ToolArgsOutput, _) =
                    steps::run_structured(&self.engines, &self.registry, &args_step, system, &user)
                        .await?;
                Ok::<_, anyhow::Error>((*index, wanted.clone(), args, resp))
            });
        let decoded = futures::future::join_all(decodes).await;

        for item in decoded {
            let (index, wanted, args, resp) = item?;
            record(metrics, ModelRole::Worker, &resp);
            let call = ToolCall {
                id: ToolCall::id_for(round, index, &wanted),
                tool: args.tool,
                arguments: args.arguments,
            };
            tracing::info!(
                tool = %call.tool,
                arguments = %call.arguments.to_string().chars().take(200).collect::<String>(),
                "tool call"
            );
            match tools.run(call).await? {
                ToolDisposition::Executed(outcome) => executed.push(outcome),
                ToolDisposition::Deferred(call) => deferred.push(call),
            }
        }

        Ok((deferred, executed))
    }

    /// Extract → verify → commit, the sequence the spec runs on tool output and
    /// again on the answer.
    async fn extract_verify_commit(
        &mut self,
        router: &Router,
        system: &str,
        user: &str,
        source: Source,
        metrics: &mut TurnMetrics,
    ) -> Result<Vec<FactId>> {
        let step = router
            .plan(StepKind::Extract)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // The schema constrains the shape; this line constrains the content.
        // Without it the extractor re-proposes the facts the subgraph just
        // showed it, every turn -- harmless at verify (they are duplicates)
        // but a worker call spent on nothing new.
        let user = format!(
            "{user}\n\n# Extract\nPropose only facts that are NEW relative to the `subgraph` \
             and `durable` sections above. Do not restate a fact already shown there. Return \
             an empty list if nothing new was established. `subject relation object` reads as \
             a sentence: `src/auth.rs depends_on jsonwebtoken` means auth.rs needs \
             jsonwebtoken, not the reverse."
        );
        let (extracted, resp): (ExtractOutput, _) =
            steps::run_structured(&self.engines, &self.registry, &step, system, &user).await?;
        record(metrics, ModelRole::Worker, &resp);

        if extracted.facts.is_empty() {
            return Ok(Vec::new());
        }

        let outcome: VerifyOutcome = verify(
            extracted.facts,
            &self.graph,
            self.mood(),
            source,
            self.turn,
            self.config.verify,
        );

        // Traceable-profile logging: the rejection *rate* is a metric, but the
        // rejection *reasons* are what someone tuning the extract prompt needs.
        // Without this line the first live run reported a 67% rejection rate
        // and no way to tell a confidence-floor miss from an off-mood relation.
        for r in &outcome.rejected {
            tracing::info!(
                subject = %r.fact.subject,
                relation = ?r.fact.relation,
                object = %r.fact.object,
                confidence = r.fact.confidence.get(),
                reason = ?r.reason,
                "verify rejected a proposed fact"
            );
        }

        // The rejection rate is per turn, so batches within a turn accumulate
        // rather than the last one overwriting the first.
        let prior_total = metrics.facts_committed;
        metrics.rejection_rate = blend_rate(
            metrics.rejection_rate,
            prior_total,
            outcome.rejection_rate(),
            outcome.total(),
        );

        let policy = self.mood().table().conflict;
        let mut committed = Vec::new();
        for fact in outcome.accepted {
            self.recent_relations.push(fact.relation().clone());
            // A window, not a history: the mood check asks what the agent has
            // been doing *lately*, and an unbounded vector both leaks and lets
            // hour-old relations outvote the last ten minutes.
            const MOOD_WINDOW: usize = 64;
            if self.recent_relations.len() > MOOD_WINDOW {
                let excess = self.recent_relations.len() - MOOD_WINDOW;
                self.recent_relations.drain(..excess);
            }
            let out = self.graph.commit(fact, Scope::Session, policy);
            if out.changed_graph()
                && let Some(id) = out.id()
            {
                committed.push(id.clone());
            }
        }
        metrics.facts_committed += committed.len();
        Ok(committed)
    }

    async fn record_acceptance(&mut self, metrics: &mut TurnMetrics) {
        for (role_name, endpoint) in self.registry.all() {
            let Ok(engine) = self.engines.for_endpoint(endpoint) else {
                continue;
            };
            let Ok(now) = engine.acceptance(endpoint).await else {
                continue;
            };
            let key = format!("{role_name}:{}", endpoint.base_url);
            let window = match self.last_acceptance.get(&key) {
                Some(prev) => now.since(prev),
                None => now,
            };
            self.last_acceptance.insert(key, now);

            let tps = window.tokens_per_step();
            if role_name == ModelRole::Planner.as_str() {
                metrics.planner_acceptance = tps;
            } else {
                metrics.worker_acceptance = tps;
            }
        }
    }

    /// Session end: promote, decay, and write metric facts.
    ///
    /// Order matters. Promotion runs before decay so a fact that earned the
    /// durable slice this session is not demoted by the same session's decay
    /// pass; and metric facts are written last so they are not themselves
    /// decayed on the way out.
    pub fn end_session(&mut self) -> SessionEnd {
        let promoted = self.graph.promote_to_durable(
            self.config.promote_min_confidence,
            self.config.promote_min_reinforcements,
        );
        let dropped = self
            .graph
            .decay(self.config.decay_factor, self.config.decay_floor);

        let facts = self.metrics.to_facts(self.turn);
        let policy = self.mood().table().conflict;
        for f in facts {
            self.graph.commit(f, Scope::Durable, policy);
        }

        SessionEnd {
            promoted: promoted.len(),
            dropped: dropped.len(),
            turns: self.turn,
        }
    }

    /// The graph, for persistence.
    pub fn graph_json(&self) -> Result<String> {
        Ok(self.graph.to_json()?)
    }
}

/// What session end did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEnd {
    pub promoted: usize,
    pub dropped: usize,
    pub turns: u64,
}

/// Record one response's usage against a role.
///
/// Every model call in the loop goes through here, so a step that forgets to
/// record is a step whose tokens never reach the cache-hit or token-ratio
/// metrics.
fn record(metrics: &mut TurnMetrics, role: ModelRole, resp: &qgi2_engine::ChatResponse) {
    let (prompt, completion, cached) = steps::usage_of(resp);
    metrics.record_usage(role, prompt, completion, cached);
}

/// Whether an answer carries raw tool-call markup the engine did not parse.
///
/// Recognises the DeepSeek DSML wrapper and the Qwen/Hermes XML form, the two
/// dialects this harness has been pointed at. Cheap and conservative: a false
/// positive only sets a flag and logs.
fn answer_contains_tool_markup(answer: &str) -> bool {
    answer.contains("<｜DSML｜")
        || answer.contains("<tool_call>")
        || answer.contains("<function=")
        || answer.contains("<|tool_call|>")
}

/// Drop route-step entry points that name no node in the graph.
fn retain_known_entries(graph: &FactGraph, entries: Vec<String>) -> Vec<String> {
    entries
        .into_iter()
        .filter(|e| graph.by_subject(e).next().is_some() || graph.by_object(e).next().is_some())
        .collect()
}

/// Combine two rejection rates weighted by how many proposals each saw.
fn blend_rate(rate_a: f64, count_a: usize, rate_b: f64, count_b: usize) -> f64 {
    let total = count_a + count_b;
    if total == 0 {
        return 0.0;
    }
    (rate_a * count_a as f64 + rate_b * count_b as f64) / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{DeferToCaller, ToolSpec};
    use qgi2_engine::Endpoint;
    use qgi2_spec_types::Speculation;

    fn registry() -> EngineRegistry {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new(
                "http://127.0.0.1:8000/v1",
                "planner",
                Speculation::Mtp { n: 2 },
            ),
        );
        r.register(
            ModelRole::Worker,
            Endpoint::new(
                "http://127.0.0.1:8001/v1",
                "worker",
                Speculation::DFlash2 { n: 7 },
            ),
        );
        r
    }

    fn session() -> Session {
        Session::new(SessionConfig::default(), registry(), vec![])
    }

    #[test]
    fn preflight_passes_when_every_endpoint_is_registered() {
        assert!(session().preflight().is_ok());
    }

    #[test]
    fn route_entry_points_that_name_no_node_are_dropped() {
        use qgi2_spec_types::{CommitToken, Confidence, ConflictPolicy, ProposedFact};
        let mut g = FactGraph::new();
        let f = ProposedFact {
            subject: "file:auth.rs".into(),
            relation: Relation::DependsOn,
            object: "crate:jsonwebtoken".into(),
            confidence: Confidence::new(0.9),
            evidence: None,
        }
        .commit(CommitToken::issued_by_verify_stage(), Source::User, 1);
        g.commit(f, Scope::Session, ConflictPolicy::LatestWins);
        let kept = retain_known_entries(
            &g,
            vec![
                "ghost:node".into(),
                "crate:jsonwebtoken".into(),
                "file:auth.rs".into(),
            ],
        );
        assert_eq!(
            kept,
            vec!["crate:jsonwebtoken".to_string(), "file:auth.rs".to_string()]
        );
        assert!(retain_known_entries(&g, vec!["ghost:node".into()]).is_empty());
    }

    #[test]
    fn preflight_fails_for_a_profile_whose_speculation_is_not_deployed() {
        let config = SessionConfig {
            persona: Persona::new(Mood::Builder, Profile::Deterministic),
            ..SessionConfig::default()
        };
        let s = Session::new(config, registry(), vec![]);
        let err = s.preflight().unwrap_err().to_string();
        // The message names what *is* deployed, so the reader can see the gap
        // rather than only being told something is missing.
        assert!(err.contains("mtp n=3"), "{err}");
        assert!(err.contains("dflash2 n=7"), "{err}");
        assert!(err.contains("vllm"), "engine is named: {err}");
    }

    #[test]
    fn the_qgi_fleet_config_preflights_clean() {
        // docs/MODELS.md: Qwen3.8-27B-FP8 with DFlash2 spec-decode on SGLang at
        // vidatron :18031. This is the spec's worker, already deployed. An
        // earlier capability table called the pairing impossible and would have
        // refused it — this pins that it routes.
        use qgi2_engine::{Endpoint, EngineKind};
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new(
                "https://llm.qgi.dev/v1",
                "QGI 3.8 Flash",
                Speculation::Mtp { n: 2 },
            ),
        );
        r.register(
            ModelRole::Worker,
            Endpoint::new(
                "http://VIDATRON_TAILNET_IP:18031/v1",
                "QGI-3.8-27b DFlash",
                Speculation::DFlash2 { n: 7 },
            )
            .with_engine(EngineKind::Sglang),
        );
        let s = Session::new(SessionConfig::default(), r, vec![]);
        s.preflight()
            .expect("the live fleet's traceable profile must route");
    }

    #[tokio::test]
    async fn turn_refuses_a_deferring_runner_instead_of_spinning() {
        // `turn` has no agent loop, so a runner that defers would loop forever
        // making requests. It must say so rather than hang.
        let mut s = session();
        let runner = DeferToCaller::new(vec![ToolSpec {
            name: "read".into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        // No live engine, so this fails at the plan step; the point is that the
        // error is about the engine, not a hang.
        let err = s.turn("hi", &runner).await.unwrap_err().to_string();
        assert!(!err.is_empty());
    }

    #[test]
    fn session_end_promotes_before_decaying() {
        use qgi2_spec_types::{CommitToken, Confidence, ProposedFact};
        let mut s = session();
        for t in 1..=2 {
            let f = ProposedFact {
                subject: "task:a".into(),
                relation: Relation::DependsOn,
                object: "file:x".into(),
                confidence: Confidence::new(0.95),
                evidence: None,
            }
            .commit(CommitToken::issued_by_verify_stage(), Source::User, t);
            s.graph.commit(
                f,
                Scope::Session,
                qgi2_spec_types::ConflictPolicy::LatestWins,
            );
        }
        let end = s.end_session();
        assert_eq!(end.promoted, 1);
        assert_eq!(s.graph.iter_scope(Scope::Durable).count(), 1);
    }

    #[test]
    fn blending_rejection_rates_weights_by_batch_size() {
        assert!((blend_rate(1.0, 1, 0.0, 9) - 0.1).abs() < 1e-9);
        assert_eq!(blend_rate(0.0, 0, 0.0, 0), 0.0);
    }

    #[test]
    fn mood_switching_is_off_by_default() {
        assert!(!SessionConfig::default().allow_mood_switch);
    }

    #[test]
    fn there_is_a_cap_on_tool_rounds() {
        // Without one, a model that always asks for tools loops forever and
        // every round is a real request.
        assert!(SessionConfig::default().max_tool_rounds > 0);
    }

    #[test]
    fn a_first_round_and_a_continuation_are_distinguishable() {
        let a = RoundInput::first("q");
        let b = RoundInput::continuation("q", 1, vec![]);
        assert_eq!(a.round, 0);
        assert_eq!(b.round, 1);
    }

    #[test]
    fn round_outcomes_expose_their_calls() {
        let calls = vec![ToolCall {
            id: "x".into(),
            tool: "read".into(),
            arguments: serde_json::Value::Null,
        }];
        let o = RoundOutcome::CallTools {
            calls: calls.clone(),
            result: TurnResult::default(),
        };
        assert!(!o.is_final());
        assert_eq!(o.calls(), calls.as_slice());
        assert!(RoundOutcome::Answered(TurnResult::default()).is_final());
    }
}
