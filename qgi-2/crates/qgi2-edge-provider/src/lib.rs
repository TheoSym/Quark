//! QGI-2 as a jcode [`Provider`].
//!
//! This is the in-process edge. It implements jcode's own provider trait, so a
//! variant binary can register it at its composition root and every jcode
//! surface — TUI, swarm, headless `run`, session resume — drives QGI-2 without
//! a single line of jcode changing.
//!
//! # Why the provider seam is the right one
//!
//! [`Provider::complete_split`] hands over `system_static` and `system_dynamic`
//! already separated, because jcode does its own prompt-cache shaping. That
//! maps onto the spec's segment model almost exactly: jcode's static half is
//! QGI-2's stable prefix (segments 1–3), its dynamic half is the volatile tail
//! (4–6). QGI-2 does not have to fight jcode's prompt construction; it refines
//! a split jcode already makes.
//!
//! # What this edge deliberately does not do
//!
//! It does not replay jcode's message history into the model. QGI-2's premise
//! is that history lives in the fact graph, so the turn loop takes the latest
//! user message and its own retrieved subgraph. The full history still exists
//! in jcode — session resume, `/resume`, the transcript all work — it just does
//! not become prompt tokens on every turn.

mod convert;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream;
use jcode_message_types::{Message, StreamEvent, ToolDefinition};
use jcode_provider_core::{EventStream, Provider};
use qgi2_engine_vllm::EngineRegistry;
use qgi2_spec_types::Persona;
use qgi2_turn::{DeferToCaller, RoundOutcome, Session, SessionConfig, session::SkillCandidate};
use std::sync::Arc;
use tokio::sync::Mutex;

pub use convert::{latest_user_text, round_input_from, text_of, tool_specs_from};

/// The name jcode knows this provider by.
///
/// Stable, because jcode keys billing and routing decisions off `Provider::name`.
pub const PROVIDER_NAME: &str = "qgi2";

/// QGI-2 behind jcode's provider trait.
///
/// The config, registry, and skill catalogue are retained alongside the live
/// session because [`Provider::fork`] must produce an instance with
/// *independent* mutable state, which means building a fresh session rather
/// than sharing this one's.
pub struct Qgi2Provider {
    session: Arc<Mutex<Session>>,
    persona: Persona,
    config: SessionConfig,
    registry: EngineRegistry,
    skills: Vec<SkillCandidate>,
}

impl Qgi2Provider {
    pub fn new(
        config: SessionConfig,
        registry: EngineRegistry,
        skills: Vec<SkillCandidate>,
    ) -> Self {
        let persona = config.persona;
        let session = Session::new(config.clone(), registry.clone(), skills.clone());
        Self {
            session: Arc::new(Mutex::new(session)),
            persona,
            config,
            registry,
            skills,
        }
    }

    pub fn with_session(session: Session) -> Self {
        let persona = session.persona();
        let config = session.config.clone();
        Self {
            session: Arc::new(Mutex::new(session)),
            persona,
            config,
            registry: EngineRegistry::new(),
            skills: Vec::new(),
        }
    }

    /// Check the engine registry before jcode starts sending turns.
    pub async fn preflight(&self) -> Result<()> {
        self.session.lock().await.preflight()
    }

    pub fn session(&self) -> Arc<Mutex<Session>> {
        self.session.clone()
    }
}

#[async_trait]
impl Provider for Qgi2Provider {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        let input = round_input_from(messages);
        if input.query.trim().is_empty() {
            // An empty turn is not an error worth failing the session over;
            // emit a well-formed empty response so jcode's loop continues.
            return Ok(Box::pin(stream::iter(vec![Ok(StreamEvent::MessageEnd {
                stop_reason: Some("stop".to_string()),
            })])));
        }

        // jcode executes tools in its own agent loop, so every call QGI-2
        // decides on is deferred back to it as a ToolUse event.
        let runner = DeferToCaller::new(tool_specs_from(tools));
        let outcome = {
            let mut session = self.session.lock().await;
            session.round(input, &runner).await?
        };

        Ok(Box::pin(stream::iter(
            events_for(&outcome).into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> String {
        format!(
            "QGI-2 ({}/{})",
            self.persona.mood.as_str(),
            self.persona.profile.as_str()
        )
    }

    fn model(&self) -> String {
        format!(
            "qgi2/{}-{}",
            self.persona.mood.as_str(),
            self.persona.profile.as_str()
        )
    }

    fn available_models_display(&self) -> Vec<String> {
        let mut out = Vec::new();
        for mood in qgi2_spec_types::Mood::ALL {
            for profile in qgi2_spec_types::Profile::ALL {
                out.push(format!("qgi2/{}-{}", mood.as_str(), profile.as_str()));
            }
        }
        out
    }

    fn set_model(&self, model: &str) -> Result<()> {
        // Switching model here means switching persona, which changes the mood
        // segment and therefore the cached prefix. Rejected rather than done
        // silently: the caller should create a session for the persona it wants
        // instead of invalidating this one's prefix mid-flight.
        Err(anyhow::anyhow!(
            "QGI-2's persona is fixed for the life of a session (asked for {model:?}). \
             A persona change rewrites the mood segment and discards the cached prefix; \
             start a session with the persona you want."
        ))
    }

    fn supports_image_input(&self) -> bool {
        false
    }

    /// A provider instance with independent mutable state.
    ///
    /// jcode forks for compaction, subagents, and resumed sessions. A fork gets
    /// its own [`Session`], not a handle to this one: sharing would let a
    /// subagent's turns advance this session's assembler prefix memory and turn
    /// the parent's next turn into a spurious `PrefixBroken`.
    ///
    /// The fork inherits a *copy* of the current graph when one can be taken
    /// without blocking. A fork requested while a turn is in flight starts from
    /// the durable slice alone; that costs the fork this session's uncommitted
    /// context but never blocks jcode's loop, and `fork` is synchronous so
    /// awaiting the lock is not available anyway.
    fn fork(&self) -> Arc<dyn Provider> {
        let mut session = Session::new(
            self.config.clone(),
            self.registry.clone(),
            self.skills.clone(),
        );
        if let Ok(live) = self.session.try_lock() {
            session = session.with_graph(live.graph.clone());
        } else {
            tracing::debug!(
                "forking QGI-2 mid-turn; the fork starts without this session's uncommitted facts"
            );
        }
        Arc::new(Self {
            session: Arc::new(Mutex::new(session)),
            persona: self.persona,
            config: self.config.clone(),
            registry: self.registry.clone(),
            skills: self.skills.clone(),
        })
    }
}

/// Turn a round outcome into the event sequence jcode's agent loop consumes.
///
/// jcode collects tool calls from `ToolUseStart` / `ToolInputDelta` /
/// `ToolUseEnd` and keeps looping while that list is non-empty
/// (`turn_loops.rs`), so this function is the whole reason QGI-2 can do
/// multi-step work through jcode rather than answering once and stopping.
pub fn events_for(outcome: &RoundOutcome) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    let stop_reason = match outcome {
        RoundOutcome::CallTools { calls, .. } => {
            for call in calls {
                events.push(StreamEvent::ToolUseStart {
                    id: call.id.clone(),
                    name: call.tool.clone(),
                });
                // jcode accumulates ToolInputDelta as a JSON *string* and parses
                // it at ToolUseEnd, so the arguments go out serialized rather
                // than as a value.
                events.push(StreamEvent::ToolInputDelta(
                    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
                ));
                events.push(StreamEvent::ToolUseEnd);
            }
            "tool_use"
        }
        RoundOutcome::Answered(result) => {
            events.push(StreamEvent::TextDelta(result.answer.clone()));
            "stop"
        }
    };

    let m = &outcome.result().metrics;
    events.push(StreamEvent::TokenUsage {
        input_tokens: Some(m.planner_prompt_tokens + m.worker_prompt_tokens),
        output_tokens: Some(m.planner_completion_tokens + m.worker_completion_tokens),
        // vLLM's real cached_tokens, so jcode's existing cache-cost warnings
        // describe QGI-2's prefix cache without knowing QGI-2 exists.
        cache_read_input_tokens: Some(m.planner_cached_tokens + m.worker_cached_tokens),
        cache_creation_input_tokens: None,
    });
    events.push(StreamEvent::MessageEnd {
        stop_reason: Some(stop_reason.to_string()),
    });

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_engine_vllm::Endpoint;
    use qgi2_spec_types::{ModelRole, Mood, Profile, Speculation};

    fn provider(persona: Persona) -> Qgi2Provider {
        let mut r = EngineRegistry::new();
        r.register(
            ModelRole::Planner,
            Endpoint::new("http://127.0.0.1:8000/v1", "p", Speculation::Mtp { n: 2 }),
        );
        r.register(
            ModelRole::Worker,
            Endpoint::new("http://127.0.0.1:8001/v1", "w", Speculation::DFlash2 { n: 7 }),
        );
        Qgi2Provider::new(
            SessionConfig {
                persona,
                ..SessionConfig::default()
            },
            r,
            vec![],
        )
    }

    #[test]
    fn the_provider_name_is_stable() {
        // jcode keys billing and routing off this.
        assert_eq!(provider(Persona::default()).name(), "qgi2");
    }

    #[test]
    fn the_display_name_shows_the_persona() {
        let p = provider(Persona::new(Mood::Researcher, Profile::Quick));
        assert_eq!(p.display_name(), "QGI-2 (researcher/quick)");
        assert_eq!(p.model(), "qgi2/researcher-quick");
    }

    #[test]
    fn switching_model_is_refused_with_a_reason() {
        let err = provider(Persona::default())
            .set_model("qgi2/companion-quick")
            .unwrap_err()
            .to_string();
        assert!(err.contains("cached prefix"), "{err}");
    }

    #[test]
    fn every_persona_is_listed() {
        assert_eq!(provider(Persona::default()).available_models_display().len(), 9);
    }

    #[tokio::test]
    async fn preflight_reports_a_missing_endpoint() {
        let p = provider(Persona::new(Mood::Builder, Profile::Deterministic));
        assert!(p.preflight().await.is_err());
    }

    fn call(id: &str, tool: &str) -> qgi2_turn::ToolCall {
        qgi2_turn::ToolCall {
            id: id.into(),
            tool: tool.into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        }
    }

    #[test]
    fn a_tool_round_emits_the_events_jcodes_loop_collects() {
        // jcode builds its tool_calls list from exactly these three events and
        // keeps looping while the list is non-empty. Without them the loop ends
        // after one pass and nothing ever runs.
        let outcome = RoundOutcome::CallTools {
            calls: vec![call("c1", "read")],
            result: qgi2_turn::TurnResult::default(),
        };
        let events = events_for(&outcome);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolUseStart { id, name } if id == "c1" && name == "read"
        ));
        assert!(matches!(&events[1], StreamEvent::ToolInputDelta(_)));
        assert!(matches!(&events[2], StreamEvent::ToolUseEnd));
    }

    #[test]
    fn tool_arguments_go_out_as_a_json_string() {
        // jcode accumulates the delta into a String and parses it at
        // ToolUseEnd; sending anything else yields an unparseable call.
        let outcome = RoundOutcome::CallTools {
            calls: vec![call("c1", "read")],
            result: qgi2_turn::TurnResult::default(),
        };
        match &events_for(&outcome)[1] {
            StreamEvent::ToolInputDelta(s) => {
                let v: serde_json::Value = serde_json::from_str(s).unwrap();
                assert_eq!(v["path"], "a.rs");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_parallel_batch_emits_a_complete_triple_per_call() {
        let outcome = RoundOutcome::CallTools {
            calls: vec![call("c1", "read"), call("c2", "ls")],
            result: qgi2_turn::TurnResult::default(),
        };
        let events = events_for(&outcome);
        let starts = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseStart { .. }))
            .count();
        let ends = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseEnd))
            .count();
        assert_eq!(starts, 2);
        assert_eq!(ends, 2, "an unmatched start leaves jcode with a partial call");
    }

    #[test]
    fn an_answer_round_emits_text_and_no_tool_events() {
        let outcome = RoundOutcome::Answered(qgi2_turn::TurnResult {
            answer: "done".into(),
            ..qgi2_turn::TurnResult::default()
        });
        let events = events_for(&outcome);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "done"));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolUseStart { .. })),
            "an answer round must not leave jcode looping"
        );
    }

    #[test]
    fn every_round_reports_usage_and_ends_the_message() {
        for outcome in [
            RoundOutcome::CallTools {
                calls: vec![call("c1", "read")],
                result: qgi2_turn::TurnResult::default(),
            },
            RoundOutcome::Answered(qgi2_turn::TurnResult::default()),
        ] {
            let events = events_for(&outcome);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, StreamEvent::TokenUsage { .. })),
                "usage must be reported on tool rounds too, or the cache metric                  only counts the last round of a multi-step turn"
            );
            assert!(matches!(
                events.last(),
                Some(StreamEvent::MessageEnd { .. })
            ));
        }
    }

    #[test]
    fn the_stop_reason_distinguishes_a_tool_round_from_an_answer() {
        let tools = events_for(&RoundOutcome::CallTools {
            calls: vec![call("c1", "read")],
            result: qgi2_turn::TurnResult::default(),
        });
        let answer = events_for(&RoundOutcome::Answered(qgi2_turn::TurnResult::default()));
        assert!(matches!(
            tools.last(),
            Some(StreamEvent::MessageEnd { stop_reason: Some(r) }) if r == "tool_use"
        ));
        assert!(matches!(
            answer.last(),
            Some(StreamEvent::MessageEnd { stop_reason: Some(r) }) if r == "stop"
        ));
    }

    #[tokio::test]
    async fn an_empty_turn_returns_a_well_formed_empty_stream() {
        use futures::StreamExt;
        let p = provider(Persona::default());
        let mut s = p.complete(&[], &[], "", None).await.unwrap();
        let first = s.next().await.unwrap().unwrap();
        assert!(matches!(first, StreamEvent::MessageEnd { .. }));
    }
}
