//! Conversions between jcode's message/tool types and QGI-2's.
//!
//! jcode's agent loop keeps the whole exchange in one message list: the user's
//! query, then assistant messages carrying [`ContentBlock::ToolUse`], then user
//! messages carrying [`ContentBlock::ToolResult`]. QGI-2 reads its round state
//! out of that list rather than tracking it, for the same reason the HTTP edge
//! does: the transcript is the conversation, and a counter kept beside it can
//! drift from it.

use jcode_message_types::{ContentBlock, Message, Role, ToolDefinition};
use qgi2_turn::{RoundInput, ToolCall, ToolOutcome, ToolSpec};

/// Convert jcode tool definitions into QGI-2 tool specs.
pub fn tool_specs_from(tools: &[ToolDefinition]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        })
        .collect()
}

/// Concatenate a message's text blocks.
///
/// Non-text blocks are skipped rather than rendered as placeholders: a
/// placeholder would become prompt bytes that change between turns for no
/// semantic reason, which is what the stable-prefix rule exists to avoid.
pub fn text_of(message: &Message) -> String {
    let mut out = String::new();
    for block in &message.content {
        if let ContentBlock::Text { text, .. } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

/// The text of the most recent user message that is a real query.
///
/// A jcode message with `Role::User` may be either the human's query or a
/// carrier for tool results. Only the former is a query, so tool-result
/// messages are skipped — otherwise the query would become empty the moment a
/// tool ran, and every continuation round would assemble a prompt with no
/// question in it.
pub fn latest_user_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::User && !carries_tool_results(m))
        .map(text_of)
        .find(|t| !t.trim().is_empty())
        .unwrap_or_default()
}

fn carries_tool_results(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
}

/// Read jcode's transcript into a round.
pub fn round_input_from(messages: &[Message]) -> RoundInput {
    let query = latest_user_text(messages);

    // Where the current turn starts: after the last real user query.
    let start = messages
        .iter()
        .rposition(|m| m.role == Role::User && !carries_tool_results(m))
        .map(|i| i + 1)
        .unwrap_or(0);

    // Map call ids to tool names, and count how many assistant tool-call
    // messages have gone by — that is the round index.
    let mut names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut round = 0u32;
    for m in &messages[start.min(messages.len())..] {
        let uses: Vec<(&String, &String)> = m
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, .. } => Some((id, name)),
                _ => None,
            })
            .collect();
        if !uses.is_empty() {
            round += 1;
            for (id, name) in uses {
                names.insert(id.clone(), name.clone());
            }
        }
    }

    // Only the trailing run of tool-result messages: earlier results were
    // already extracted on the round that received them, and re-extracting
    // would double-count facts and inflate the reinforcement counts that gate
    // promotion to the durable slice.
    let trailing = messages
        .iter()
        .rev()
        .take_while(|m| carries_tool_results(m))
        .count();

    let mut tool_results = Vec::new();
    for m in &messages[messages.len() - trailing..] {
        for block in &m.content {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } = block
            {
                let tool = names
                    .get(tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let call = ToolCall {
                    id: tool_use_id.clone(),
                    tool,
                    arguments: serde_json::Value::Null,
                };
                tool_results.push(if is_error.unwrap_or(false) {
                    ToolOutcome::error(call, content.clone())
                } else {
                    ToolOutcome::ok(call, content.clone())
                });
            }
        }
    }

    RoundInput {
        query,
        tool_results,
        round,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: Vec<ContentBlock>) -> Message {
        Message {
            role,
            content,
            timestamp: None,
            tool_duration_ms: None,
        }
    }

    fn text(t: &str) -> ContentBlock {
        ContentBlock::Text {
            text: t.into(),
            cache_control: None,
        }
    }

    fn user(t: &str) -> Message {
        msg(Role::User, vec![text(t)])
    }

    fn tool_use(id: &str, name: &str) -> Message {
        msg(
            Role::Assistant,
            vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }],
        )
    }

    fn tool_result(id: &str, content: &str, is_error: bool) -> Message {
        msg(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: content.into(),
                is_error: is_error.then_some(true),
            }],
        )
    }

    #[test]
    fn a_fresh_query_is_round_zero() {
        let r = round_input_from(&[user("read main.rs")]);
        assert_eq!(r.query, "read main.rs");
        assert_eq!(r.round, 0);
        assert!(r.tool_results.is_empty());
    }

    #[test]
    fn a_tool_result_message_does_not_erase_the_query() {
        // jcode carries tool results on a Role::User message. Treating that as
        // the query would leave every continuation round with no question.
        let msgs = vec![
            user("read main.rs"),
            tool_use("c1", "read"),
            tool_result("c1", "fn main() {}", false),
        ];
        let r = round_input_from(&msgs);
        assert_eq!(r.query, "read main.rs");
        assert_eq!(r.round, 1);
        assert_eq!(r.tool_results.len(), 1);
        assert_eq!(r.tool_results[0].output, "fn main() {}");
    }

    #[test]
    fn a_result_is_matched_back_to_its_tools_name() {
        let msgs = vec![
            user("q"),
            tool_use("c1", "agentgrep"),
            tool_result("c1", "hits", false),
        ];
        let r = round_input_from(&msgs);
        assert_eq!(r.tool_results[0].call.tool, "agentgrep");
    }

    #[test]
    fn tool_failures_survive_as_failures() {
        let msgs = vec![
            user("q"),
            tool_use("c1", "read"),
            tool_result("c1", "no such file", true),
        ];
        let r = round_input_from(&msgs);
        assert!(r.tool_results[0].is_error);
        assert!(r.tool_results[0].render().contains("[FAILED]"));
    }

    #[test]
    fn several_rounds_accumulate() {
        let msgs = vec![
            user("q"),
            tool_use("c1", "ls"),
            tool_result("c1", "a.rs", false),
            tool_use("c2", "read"),
            tool_result("c2", "contents", false),
        ];
        assert_eq!(round_input_from(&msgs).round, 2);
    }

    #[test]
    fn only_the_trailing_results_are_returned() {
        let msgs = vec![
            user("q"),
            tool_use("c1", "ls"),
            tool_result("c1", "old", false),
            tool_use("c2", "read"),
            tool_result("c2", "new", false),
        ];
        let r = round_input_from(&msgs);
        assert_eq!(r.tool_results.len(), 1);
        assert_eq!(r.tool_results[0].output, "new");
    }

    #[test]
    fn a_new_query_resets_the_round() {
        let msgs = vec![
            user("first"),
            tool_use("c1", "ls"),
            tool_result("c1", "out", false),
            user("second"),
        ];
        let r = round_input_from(&msgs);
        assert_eq!(r.query, "second");
        assert_eq!(r.round, 0);
        assert!(r.tool_results.is_empty());
    }

    #[test]
    fn a_batch_of_results_in_one_message_is_read_whole() {
        // jcode may pack several ToolResult blocks into one message.
        let msgs = vec![
            user("q"),
            msg(
                Role::Assistant,
                vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "read".into(),
                        input: serde_json::json!({}),
                        thought_signature: None,
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "ls".into(),
                        input: serde_json::json!({}),
                        thought_signature: None,
                    },
                ],
            ),
            msg(
                Role::User,
                vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "c1".into(),
                        content: "one".into(),
                        is_error: None,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "c2".into(),
                        content: "two".into(),
                        is_error: None,
                    },
                ],
            ),
        ];
        let r = round_input_from(&msgs);
        assert_eq!(r.tool_results.len(), 2);
        assert_eq!(r.round, 1);
    }

    #[test]
    fn an_empty_history_yields_an_empty_query() {
        assert!(round_input_from(&[]).query.is_empty());
    }

    #[test]
    fn multiple_text_blocks_are_joined() {
        let m = msg(Role::User, vec![text("one"), text("two")]);
        assert_eq!(text_of(&m), "one\ntwo");
    }

    #[test]
    fn non_text_blocks_are_skipped_rather_than_placeheld() {
        let m = msg(
            Role::User,
            vec![
                ContentBlock::Reasoning {
                    text: "hidden".into(),
                },
                text("visible"),
            ],
        );
        assert_eq!(text_of(&m), "visible");
    }
}
