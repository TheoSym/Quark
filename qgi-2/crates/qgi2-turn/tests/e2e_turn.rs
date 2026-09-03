//! The turn loop, end to end, with no model and no server.
//!
//! Until these existed, `Session::round` -- the most consequential code in the
//! project -- had never executed anywhere: every model step went over HTTP, so
//! it could only be exercised against a live deployment that did not exist.
//! These tests drive the real loop through the real router, assembler, rules,
//! graph and metrics; only the engine is scripted.

use qgi2_engine::{Endpoint, EngineKind, EngineRegistry, Script, ScriptedEngine, SeenStep};
use qgi2_metrics::BreachKind;
use qgi2_spec_types::{Mood, ModelRole, Persona, Profile, Speculation};
use qgi2_turn::steps::Engines;
use qgi2_turn::tools::ToolSpec;
use qgi2_turn::{DeferToCaller, RoundInput, RoundOutcome, Session, SessionConfig, ToolOutcome};
use std::sync::Arc;

fn registry() -> EngineRegistry {
    let mut r = EngineRegistry::new();
    r.register(
        ModelRole::Planner,
        Endpoint::new("http://scripted/v1", "planner", Speculation::Mtp { n: 2 })
            .with_engine(EngineKind::Sglang),
    );
    r.register(
        ModelRole::Worker,
        Endpoint::new("http://scripted/v1", "worker", Speculation::DFlash2 { n: 7 })
            .with_engine(EngineKind::Sglang),
    );
    r.set_embedder(
        Endpoint::new("http://scripted/v1", "embed", Speculation::Off).with_engine(EngineKind::Sglang),
    );
    r
}

/// A session whose every model call answers from `script`.
fn session_with(script: Script, config: SessionConfig) -> (Session, Arc<ScriptedEngine>) {
    let engine = Arc::new(ScriptedEngine::new(script));
    let reg = registry();
    let engines = Engines::for_registry(&reg).with_engine(EngineKind::Sglang, engine.clone());
    let session = Session::new(config, reg, vec![]).with_engines(engines);
    (session, engine)
}

fn tools() -> DeferToCaller {
    DeferToCaller::new(vec![
        ToolSpec {
            name: "read".into(),
            description: "read a file".into(),
            parameters: serde_json::json!({
                "type": "object", "required": ["path"],
                "properties": { "path": { "type": "string" } }
            }),
        },
        ToolSpec {
            name: "gmail".into(),
            description: "send mail".into(),
            parameters: serde_json::json!({ "type": "object" }),
        },
    ])
}

#[tokio::test]
async fn a_turn_with_no_tools_answers_and_commits_facts() {
    let script = Script {
        plans: vec![Script::plan_answering()],
        extracts: vec![Script::extracting(&[("task:auth", "depends_on", "file:auth.rs", 0.9)])],
        answers: vec!["The auth task depends on auth.rs.".into()],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());

    let out = s.round(RoundInput::first("what does auth need?"), &tools()).await.unwrap();

    let RoundOutcome::Answered(r) = out else { panic!("expected an answer") };
    assert_eq!(r.answer, "The auth task depends on auth.rs.");
    assert_eq!(r.committed.len(), 1, "the extracted fact reached the graph");
    assert_eq!(s.graph.len(), 1);
    assert_eq!(s.turn_index(), 1);

    // The spec's loop, minus the tool phase and minus route (graph was empty).
    assert_eq!(
        engine.steps_called(),
        vec![SeenStep::Plan, SeenStep::Answer, SeenStep::Extract]
    );
}

#[tokio::test]
async fn the_route_step_runs_once_the_graph_has_something_to_route_through() {
    let script = Script {
        plans: vec![Script::plan_answering()],
        extracts: vec![Script::extracting(&[("task:auth", "depends_on", "file:auth.rs", 0.9)])],
        routes: vec![serde_json::json!({ "entry_points": ["task:auth"] })],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());

    s.round(RoundInput::first("first"), &tools()).await.unwrap();
    assert!(!engine.steps_called().contains(&SeenStep::Route), "nothing to route yet");

    s.round(RoundInput::first("second"), &tools()).await.unwrap();
    assert!(
        engine.steps_called().contains(&SeenStep::Route),
        "with facts in the graph, the worker refines entry points: {:?}",
        engine.steps_called()
    );
}

#[tokio::test]
async fn a_tool_round_hands_the_call_back_and_the_continuation_finishes_the_turn() {
    let script = Script {
        plans: vec![
            Script::plan_calling("read", "read the auth module"),
            Script::plan_answering(),
        ],
        tool_args: vec![serde_json::json!({ "tool": "read", "arguments": { "path": "auth.rs" } })],
        extracts: vec![Script::extracting(&[("file:auth.rs", "part_of", "module:auth", 0.8)])],
        answers: vec!["auth.rs is part of the auth module.".into()],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());

    // Round 0: the loop decides on a tool and hands it back.
    let out = s.round(RoundInput::first("read auth"), &tools()).await.unwrap();
    let RoundOutcome::CallTools { calls, result } = out else { panic!("expected a tool call") };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool, "read");
    assert_eq!(calls[0].arguments["path"], "auth.rs");
    assert_eq!(calls[0].id, "qgi2-0-0-read", "deterministic id");
    assert!(result.answer.is_empty(), "no answer until the final round");

    // Round 1: the caller ran it; the loop extracts from the result, then answers.
    let results = vec![ToolOutcome::ok(calls[0].clone(), "fn login() {}")];
    let out = s
        .round(RoundInput::continuation("read auth", 1, results), &tools())
        .await
        .unwrap();
    let RoundOutcome::Answered(r) = out else { panic!("expected an answer") };
    assert_eq!(r.answer, "auth.rs is part of the auth module.");
    assert!(!s.graph.is_empty(), "facts from the tool result were committed");

    // One turn, not two: rounds accumulate into it.
    assert_eq!(s.turn_index(), 1);
    assert_eq!(s.metrics.turns.len(), 1);

    // The tool result reached the extract step verbatim.
    let extract_prompt = engine
        .seen()
        .into_iter()
        .find(|x| x.step == SeenStep::Extract)
        .expect("an extract call")
        .user;
    assert!(extract_prompt.contains("fn login() {}"), "{extract_prompt}");
}

#[tokio::test]
async fn the_round_cap_forces_an_answer_and_tells_the_model() {
    let script = Script {
        // Always wants a tool, never answers on its own.
        plans: vec![Script::plan_calling("read", "keep reading")],
        answers: vec!["I ran out of tool budget.".into()],
        ..Script::default()
    };
    let config = SessionConfig {
        max_tool_rounds: 1,
        ..SessionConfig::default()
    };
    let (mut s, engine) = session_with(script, config);

    let out = s.round(RoundInput::first("go"), &tools()).await.unwrap();
    assert!(matches!(out, RoundOutcome::CallTools { .. }), "round 0 may call");

    let out = s
        .round(RoundInput::continuation("go", 1, vec![]), &tools())
        .await
        .unwrap();
    let RoundOutcome::Answered(r) = out else { panic!("cap must force an answer") };
    assert!(r.tool_rounds_exhausted);

    // The flag alone informed the caller; the model has to be told too, or it
    // answers as if the work finished.
    let answer_prompt = engine
        .seen()
        .into_iter()
        .find(|x| x.step == SeenStep::Answer)
        .expect("an answer call")
        .user;
    assert!(answer_prompt.contains("budget"), "{answer_prompt}");
}

#[tokio::test]
async fn a_tool_outside_the_mood_is_refused_not_called() {
    let script = Script {
        plans: vec![
            Script::plan_calling("gmail", "send the report"),
            Script::plan_answering(),
        ],
        ..Script::default()
    };
    // Builder admits fs/shell/git; gmail is Companion's.
    let (mut s, engine) = session_with(script, SessionConfig::default());

    let out = s.round(RoundInput::first("email it"), &tools()).await.unwrap();
    // Nothing was deferred: the masked tool became an inline error result and
    // the loop went on to answer.
    let RoundOutcome::Answered(r) = out else { panic!("a masked tool must not be handed back") };
    assert_eq!(r.tools.len(), 1);
    assert!(r.tools[0].is_error);
    assert!(r.tools[0].output.contains("mood"), "{}", r.tools[0].output);
    // And no tool-args call was spent deciding arguments for a tool that
    // cannot run.
    assert!(!engine.steps_called().contains(&SeenStep::ToolArgs));
}

#[tokio::test]
async fn verify_rejections_reach_the_rejection_rate_metric() {
    let script = Script {
        plans: vec![Script::plan_answering()],
        extracts: vec![Script::extracting(&[
            ("task:a", "depends_on", "file:x", 0.9), // accepted
            ("task:b", "depends_on", "file:y", 0.05), // below the floor
        ])],
        ..Script::default()
    };
    let (mut s, _) = session_with(script, SessionConfig::default());

    let out = s.round(RoundInput::first("q"), &tools()).await.unwrap();
    let RoundOutcome::Answered(r) = out else { panic!() };
    assert_eq!(r.committed.len(), 1);
    assert!((r.metrics.rejection_rate - 0.5).abs() < 1e-9);
    assert!(
        r.breaches.iter().any(|b| b.kind == BreachKind::RejectionRate),
        "50% rejection is over the 10% ceiling: {:?}",
        r.breaches
    );
}

#[tokio::test]
async fn cache_hit_rate_comes_from_the_engines_usage() {
    let good = Script {
        prompt_tokens: 1000,
        cached_tokens: 950,
        ..Script::default()
    };
    let (mut s, _) = session_with(good, SessionConfig::default());
    let RoundOutcome::Answered(r) = s.round(RoundInput::first("q"), &tools()).await.unwrap() else { panic!() };
    assert!(!r.breaches.iter().any(|b| b.kind == BreachKind::CacheHitRate));
    assert_eq!(r.metrics.cache_hit_rate(ModelRole::Planner), Some(0.95));

    let bad = Script {
        prompt_tokens: 1000,
        cached_tokens: 100,
        ..Script::default()
    };
    let (mut s, _) = session_with(bad, SessionConfig::default());
    let RoundOutcome::Answered(r) = s.round(RoundInput::first("q"), &tools()).await.unwrap() else { panic!() };
    assert!(
        r.breaches.iter().any(|b| b.kind == BreachKind::CacheHitRate),
        "10% cache is a breach, and the spec calls it a bug"
    );
}

#[tokio::test]
async fn a_down_embedder_degrades_to_lexical_and_says_so() {
    let script = Script {
        embedder_up: false,
        ..Script::default()
    };
    let (mut s, _) = session_with(script, SessionConfig::default());
    let RoundOutcome::Answered(r) = s.round(RoundInput::first("q"), &tools()).await.unwrap() else { panic!() };
    assert!(r.retrieval_degraded, "a Traceable run at Quick-grade retrieval must be visible");
    assert!(!r.answer.is_empty(), "but the turn still completes");
}

#[tokio::test]
async fn the_quick_profile_never_calls_the_embedder_or_the_route_step() {
    let script = Script {
        extracts: vec![Script::extracting(&[("task:a", "depends_on", "file:x", 0.9)])],
        ..Script::default()
    };
    let config = SessionConfig {
        persona: Persona::new(Mood::Builder, Profile::Quick),
        ..SessionConfig::default()
    };
    let (mut s, engine) = session_with(script, config);
    s.round(RoundInput::first("first"), &tools()).await.unwrap();
    s.round(RoundInput::first("second"), &tools()).await.unwrap();

    assert!(engine.embed_calls().is_empty(), "Quick is lexical-only");
    assert!(!engine.steps_called().contains(&SeenStep::Route));
    let RoundOutcome::Answered(r) = s.round(RoundInput::first("third"), &tools()).await.unwrap() else { panic!() };
    assert!(!r.retrieval_degraded, "not degraded -- Quick never wanted embeddings");
}

#[tokio::test]
async fn new_subjects_are_embedded_after_commit_not_before() {
    let script = Script {
        extracts: vec![Script::extracting(&[
            ("task:a", "depends_on", "file:x", 0.9),
            ("task:junk", "depends_on", "file:y", 0.01), // rejected at verify
        ])],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());
    s.round(RoundInput::first("q"), &tools()).await.unwrap();

    let embedded: Vec<String> = engine.embed_calls().into_iter().flatten().collect();
    assert!(embedded.contains(&"task:a".to_string()));
    assert!(
        !embedded.contains(&"task:junk".to_string()),
        "a rejected proposal must never earn an embedder call"
    );
}

#[tokio::test]
async fn every_model_step_is_streamed_and_capped() {
    let (mut s, engine) = session_with(Script::default(), SessionConfig::default());
    s.round(RoundInput::first("q"), &tools()).await.unwrap();
    for seen in engine.seen() {
        assert!(seen.streamed, "{:?} was not streamed", seen.step);
        assert!(seen.max_tokens.is_some(), "{:?} had no token cap", seen.step);
    }
}

#[tokio::test]
async fn worker_steps_run_cooler_than_the_planner_wants() {
    // Thinking is off on the worker; sampling still comes from the mood.
    let script = Script {
        plans: vec![Script::plan_calling("read", "look"), Script::plan_answering()],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());
    s.round(RoundInput::first("q"), &tools()).await.unwrap();
    let seen = engine.seen();
    let plan = seen.iter().find(|x| x.step == SeenStep::Plan).unwrap();
    let args = seen.iter().find(|x| x.step == SeenStep::ToolArgs).unwrap();
    assert_eq!(plan.model, "planner");
    assert_eq!(args.model, "worker", "tool args route to the worker");
    assert_eq!(plan.temperature, 0.3, "Builder mood");
    assert_eq!(args.temperature, 0.3);
}

#[tokio::test]
async fn the_stable_prefix_survives_across_turns() {
    let (mut s, engine) = session_with(Script::default(), SessionConfig::default());
    s.round(RoundInput::first("first question"), &tools()).await.unwrap();
    s.round(RoundInput::first("a completely different question"), &tools()).await.unwrap();

    // Both plan calls carried byte-identical system prompts: that is the
    // whole premise, and the thing the cache metric is measuring.
    let systems: Vec<String> = engine
        .seen()
        .into_iter()
        .filter(|x| x.step == SeenStep::Plan)
        .map(|x| x.system)
        .collect();
    assert_eq!(systems.len(), 2);
    assert_eq!(systems[0], systems[1]);
}


// ---------- per-process optimisations ----------

#[tokio::test]
async fn the_query_is_embedded_once_per_turn_not_once_per_round() {
    // A two-round turn: tool call, then answer. The query is the same string
    // in both rounds, so a second embedder call would return the same vector.
    let script = Script {
        plans: vec![
            Script::plan_calling("read", "read it"),
            Script::plan_answering(),
        ],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());

    let out = s.round(RoundInput::first("read auth"), &tools()).await.unwrap();
    let RoundOutcome::CallTools { calls, .. } = out else { panic!() };
    let results = vec![ToolOutcome::ok(calls[0].clone(), "fn login() {}")];
    s.round(RoundInput::continuation("read auth", 1, results), &tools())
        .await
        .unwrap();

    let query_embeds = engine
        .embed_calls()
        .into_iter()
        .filter(|batch| batch == &vec!["read auth".to_string()])
        .count();
    assert_eq!(query_embeds, 1, "the query vector is computed once per turn");

    // And a new turn embeds again -- the cache is per turn, not per session.
    s.round(RoundInput::first("read auth"), &tools()).await.unwrap();
    let query_embeds = engine
        .embed_calls()
        .into_iter()
        .filter(|batch| batch == &vec!["read auth".to_string()])
        .count();
    assert_eq!(query_embeds, 2);
}

#[tokio::test]
async fn a_multi_tool_plan_produces_every_call() {
    // Argument decodes are issued concurrently and joined; the observable
    // contract is that all of them happen and come back in plan order.
    let script = Script {
        plans: vec![serde_json::json!({
            "needs_tools": true,
            "steps": [
                { "intent": "read a", "tool": "read" },
                { "intent": "read b", "tool": "read" },
                { "intent": "read c", "tool": "read" }
            ]
        })],
        tool_args: vec![
            serde_json::json!({ "tool": "read", "arguments": { "path": "a.rs" } }),
            serde_json::json!({ "tool": "read", "arguments": { "path": "b.rs" } }),
            serde_json::json!({ "tool": "read", "arguments": { "path": "c.rs" } }),
        ],
        ..Script::default()
    };
    let (mut s, engine) = session_with(script, SessionConfig::default());

    let out = s.round(RoundInput::first("read all three"), &tools()).await.unwrap();
    let RoundOutcome::CallTools { calls, .. } = out else { panic!() };
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        vec!["qgi2-0-0-read", "qgi2-0-1-read", "qgi2-0-2-read"],
        "plan order is preserved whatever order the decodes finished in"
    );
    assert_eq!(
        engine.steps_called().iter().filter(|s| **s == SeenStep::ToolArgs).count(),
        3
    );
}

#[tokio::test]
async fn oversized_tool_output_is_windowed_before_it_reaches_the_model() {
    let script = Script {
        plans: vec![
            Script::plan_calling("read", "read the big file"),
            Script::plan_answering(),
        ],
        ..Script::default()
    };
    let config = SessionConfig {
        max_tool_output_bytes: 200,
        ..SessionConfig::default()
    };
    let (mut s, engine) = session_with(script, config);

    let out = s.round(RoundInput::first("read big"), &tools()).await.unwrap();
    let RoundOutcome::CallTools { calls, .. } = out else { panic!() };
    let big = format!("HEAD{}TAIL", "x".repeat(5000));
    let results = vec![ToolOutcome::ok(calls[0].clone(), big)];
    let out = s
        .round(RoundInput::continuation("read big", 1, results), &tools())
        .await
        .unwrap();
    let RoundOutcome::Answered(r) = out else { panic!() };
    assert_eq!(r.tool_outputs_truncated, 1);

    for step in [SeenStep::Extract, SeenStep::Answer] {
        let prompt = engine
            .seen()
            .into_iter()
            .find(|x| x.step == step)
            .unwrap_or_else(|| panic!("no {step:?} call"))
            .user;
        assert!(prompt.len() < 2000, "{step:?} prompt was {} bytes", prompt.len());
        assert!(prompt.contains("bytes omitted"), "{step:?} prompt does not say what it dropped");
        assert!(prompt.contains("HEAD"), "{step:?}: head kept");
        assert!(prompt.contains("TAIL"), "{step:?}: tail kept");
    }
}

#[tokio::test]
async fn the_route_step_can_be_skipped_on_an_exact_key_hit() {
    let script = Script {
        plans: vec![Script::plan_answering()],
        extracts: vec![Script::extracting(&[("task:auth", "depends_on", "file:auth.rs", 0.9)])],
        ..Script::default()
    };

    // Default: route runs even when the user named the node.
    let (mut s, engine) = session_with(script.clone(), SessionConfig::default());
    s.round(RoundInput::first("seed"), &tools()).await.unwrap();
    s.round(RoundInput::first("tell me about task:auth"), &tools()).await.unwrap();
    assert!(engine.steps_called().contains(&SeenStep::Route));

    // Opted in: an exact-key hit makes the worker call redundant.
    let config = SessionConfig {
        skip_route_on_exact_hit: true,
        ..SessionConfig::default()
    };
    let (mut s, engine) = session_with(script, config);
    s.round(RoundInput::first("seed"), &tools()).await.unwrap();
    s.round(RoundInput::first("tell me about task:auth"), &tools()).await.unwrap();
    assert!(!engine.steps_called().contains(&SeenStep::Route));

    // ...but a query with no exact hit still routes.
    s.round(RoundInput::first("what depends on what?"), &tools()).await.unwrap();
    assert!(engine.steps_called().contains(&SeenStep::Route));
}
