//! The OpenAI function-calling wire contract.
//!
//! This is what makes the sidecar usable for real work rather than
//! conversation. jcode sends its tool catalogue in `tools`; QGI-2 answers with
//! `tool_calls` and `finish_reason: "tool_calls"`; jcode executes them and
//! sends the results back as `role: "tool"` messages carrying `tool_call_id`.
//! Each of those exchanges is one QGI-2 *round*, and the whole sequence is one
//! *turn*.
//!
//! The round index is recovered from the transcript rather than tracked
//! server-side. An OpenAI-compatible endpoint has no session identity — jcode
//! may reconnect, retry, or run several conversations against one server — so
//! any counter the server kept would drift from the conversation it describes.
//! Counting assistant tool-call messages in the transcript cannot drift,
//! because the transcript *is* the conversation.

use qgi2_turn::{ToolCall, ToolOutcome, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool as OpenAI clients declare it.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ToolDeclaration {
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionDeclaration,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Absent for a no-argument tool; an empty object schema is the right
    /// reading, and rejecting the tool instead would drop it from the mask.
    #[serde(default)]
    pub parameters: Option<Value>,
}

impl ToolDeclaration {
    pub fn to_spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.function.name.clone(),
            description: self.function.description.clone(),
            parameters: self
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} })),
        }
    }
}

/// One message in the transcript.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    /// Null on an assistant message that only carries tool calls.
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    /// Present on `role: "tool"` messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Some clients send the tool's name alongside the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallMessage {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments. A *string*, per the OpenAI contract — not an
    /// object. Emitting an object here is the single most common way to break
    /// a client that follows the spec.
    pub arguments: String,
}

impl ToolCallMessage {
    pub fn from_call(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            kind: function_type(),
            function: FunctionCall {
                name: call.tool.clone(),
                arguments: serde_json::to_string(&call.arguments)
                    .unwrap_or_else(|_| "{}".to_string()),
            },
        }
    }
}

/// What the transcript says about where this turn stands.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptState {
    /// The user's query for the turn in flight.
    pub query: String,
    /// Round index: how many assistant tool-call messages precede the tail.
    pub round: u32,
    /// Results for the calls the previous round emitted.
    pub tool_results: Vec<ToolOutcome>,
}

/// Read the transcript.
///
/// The query is the last `user` message. Results are the trailing run of
/// `tool` messages — trailing, because only those answer the most recent batch
/// of calls; earlier ones were already folded into the graph on the round that
/// received them, and re-extracting them would double-count facts and inflate
/// the reinforcement counts that gate promotion to the durable slice.
pub fn read_transcript(messages: &[Message]) -> TranscriptState {
    let last_user = messages.iter().rposition(|m| m.role == "user");
    let query = last_user
        .map(|i| messages[i].text().to_string())
        .unwrap_or_default();

    // Calls made since that user message, so a call id can be matched to a name.
    let after = last_user.map(|i| i + 1).unwrap_or(0);
    let mut names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut round = 0u32;
    for m in &messages[after.min(messages.len())..] {
        if m.role == "assistant"
            && let Some(calls) = &m.tool_calls
            && !calls.is_empty()
        {
            round += 1;
            for c in calls {
                names.insert(c.id.clone(), c.function.name.clone());
            }
        }
    }

    let trailing = messages
        .iter()
        .rev()
        .take_while(|m| m.role == "tool")
        .count();
    let tool_results = messages[messages.len() - trailing..]
        .iter()
        .map(|m| {
            let id = m.tool_call_id.clone().unwrap_or_default();
            let tool = m
                .name
                .clone()
                .or_else(|| names.get(&id).cloned())
                .unwrap_or_else(|| "unknown".to_string());
            let text = m.text().to_string();
            // Clients do not mark tool failures in a standard field, so the
            // harness records the output verbatim and lets the model read it.
            // Guessing at failure from the text would mislabel a successful
            // grep for the word "error".
            ToolOutcome::ok(
                ToolCall {
                    id,
                    tool,
                    arguments: Value::Null,
                },
                text,
            )
        })
        .collect();

    TranscriptState {
        query,
        round,
        tool_results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message {
            role: "user".into(),
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant_calls(calls: &[(&str, &str)]) -> Message {
        Message {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(
                calls
                    .iter()
                    .map(|(id, name)| ToolCallMessage {
                        id: (*id).into(),
                        kind: "function".into(),
                        function: FunctionCall {
                            name: (*name).into(),
                            arguments: "{}".into(),
                        },
                    })
                    .collect(),
            ),
            tool_call_id: None,
            name: None,
        }
    }

    fn tool_result(id: &str, text: &str) -> Message {
        Message {
            role: "tool".into(),
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: Some(id.into()),
            name: None,
        }
    }

    #[test]
    fn a_fresh_query_is_round_zero_with_no_results() {
        let s = read_transcript(&[user("read main.rs")]);
        assert_eq!(s.query, "read main.rs");
        assert_eq!(s.round, 0);
        assert!(s.tool_results.is_empty());
    }

    #[test]
    fn a_tool_result_advances_the_round_and_carries_the_output() {
        let msgs = vec![
            user("read main.rs"),
            assistant_calls(&[("c1", "read")]),
            tool_result("c1", "fn main() {}"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.query, "read main.rs");
        assert_eq!(s.round, 1);
        assert_eq!(s.tool_results.len(), 1);
        assert_eq!(s.tool_results[0].output, "fn main() {}");
    }

    #[test]
    fn a_result_is_matched_back_to_its_tools_name_by_id() {
        let msgs = vec![
            user("q"),
            assistant_calls(&[("c1", "agentgrep")]),
            tool_result("c1", "hits"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.tool_results[0].call.tool, "agentgrep");
    }

    #[test]
    fn several_rounds_accumulate() {
        let msgs = vec![
            user("q"),
            assistant_calls(&[("c1", "ls")]),
            tool_result("c1", "a.rs"),
            assistant_calls(&[("c2", "read")]),
            tool_result("c2", "contents"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.round, 2);
    }

    #[test]
    fn only_the_trailing_run_of_results_is_returned() {
        // Earlier results were already extracted on the round that received
        // them; re-extracting would double-count facts and inflate the
        // reinforcement counts that gate promotion to the durable slice.
        let msgs = vec![
            user("q"),
            assistant_calls(&[("c1", "ls")]),
            tool_result("c1", "old"),
            assistant_calls(&[("c2", "read")]),
            tool_result("c2", "new"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.tool_results.len(), 1);
        assert_eq!(s.tool_results[0].output, "new");
    }

    #[test]
    fn a_parallel_batch_returns_every_result() {
        let msgs = vec![
            user("q"),
            assistant_calls(&[("c1", "read"), ("c2", "ls")]),
            tool_result("c1", "one"),
            tool_result("c2", "two"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.tool_results.len(), 2);
        assert_eq!(s.round, 1);
    }

    #[test]
    fn a_new_user_message_resets_the_round() {
        // The next turn starts at round 0 even though the transcript is long.
        let msgs = vec![
            user("first"),
            assistant_calls(&[("c1", "ls")]),
            tool_result("c1", "out"),
            user("second"),
        ];
        let s = read_transcript(&msgs);
        assert_eq!(s.query, "second");
        assert_eq!(s.round, 0);
        assert!(s.tool_results.is_empty());
    }

    #[test]
    fn arguments_serialize_as_a_json_string_not_an_object() {
        // The single most common way to break a spec-following client.
        let call = ToolCall {
            id: "c1".into(),
            tool: "read".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        };
        let msg = ToolCallMessage::from_call(&call);
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v["function"]["arguments"].is_string(), "{v}");
        assert_eq!(v["function"]["arguments"], "{\"path\":\"a.rs\"}");
    }

    #[test]
    fn a_tool_with_no_parameters_still_produces_a_usable_schema() {
        let d = ToolDeclaration {
            kind: "function".into(),
            function: FunctionDeclaration {
                name: "pwd".into(),
                description: String::new(),
                parameters: None,
            },
        };
        assert_eq!(d.to_spec().parameters["type"], "object");
    }

    #[test]
    fn an_assistant_message_with_null_content_deserializes() {
        // OpenAI sends content: null on a pure tool-call message.
        let m: Message = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}]}"#,
        )
        .unwrap();
        assert_eq!(m.text(), "");
        assert_eq!(m.tool_calls.unwrap().len(), 1);
    }
}
