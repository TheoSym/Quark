//! JSON schemas for the structured steps.
//!
//! Spec invariant:
//!
//! > Every structured step (extract, render, tool-args, route) runs under a
//! > JSON schema. No free-text bookkeeping.
//!
//! These are handed to vLLM as `guided_json`, so the engine constrains decoding
//! to the grammar rather than the harness parsing hopefully-valid JSON
//! afterwards. That is also what makes speculation pay: a constrained decode
//! has far fewer plausible next tokens, so more speculative tokens are accepted.
//!
//! Every schema sets `additionalProperties: false`. Without it a model can
//! append plausible-looking extra keys that the harness silently drops, which is
//! precisely the free-text bookkeeping the invariant rules out.

use qgi2_spec_types::StepKind;
use serde_json::{Value, json};

/// The schema for a step, or `None` for the free-text answer step.
pub fn for_step(step: StepKind) -> Option<Value> {
    match step {
        StepKind::Plan => Some(plan_schema()),
        StepKind::Extract => Some(extract_schema()),
        StepKind::ToolArgs => Some(tool_args_schema()),
        StepKind::Route => Some(route_schema()),
        StepKind::Answer | StepKind::Verify | StepKind::Commit | StepKind::MoodCheck => None,
    }
}

/// Relation names the extractor may use.
///
/// Enumerated in the schema rather than accepted as free strings so the model
/// cannot invent a relation the mood rules have never heard of. An extraction
/// with a made-up relation would be rejected at verify anyway; constraining it
/// here means the tokens are never spent.
fn relation_enum() -> Value {
    json!([
        "depends_on",
        "implements",
        "modifies",
        "supports",
        "contradicts",
        "cited_by",
        "prefers",
        "dislikes",
        "knows_about",
        "is_a",
        "part_of"
    ])
}

fn plan_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["steps", "needs_tools"],
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["intent"],
                    "properties": {
                        "intent": { "type": "string", "maxLength": 200 },
                        "tool": { "type": "string", "maxLength": 64 }
                    }
                }
            },
            "needs_tools": { "type": "boolean" }
        }
    })
}

fn extract_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["facts"],
        "properties": {
            "facts": {
                "type": "array",
                // A cap belongs in the schema, not in post-processing: without
                // it a model can spend an entire generation budget emitting
                // low-value facts that verify then throws away.
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["subject", "relation", "object", "confidence"],
                    "properties": {
                        "subject": { "type": "string", "minLength": 1, "maxLength": 128 },
                        "relation": { "type": "string", "enum": relation_enum() },
                        "object": { "type": "string", "minLength": 1, "maxLength": 128 },
                        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                        "evidence": { "type": "string", "maxLength": 300 }
                    }
                }
            }
        }
    })
}

fn tool_args_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["tool", "arguments"],
        "properties": {
            "tool": { "type": "string", "minLength": 1, "maxLength": 64 },
            // The tool's own parameter schema is spliced in by
            // `tool_args_schema_for`; this is the envelope.
            "arguments": { "type": "object" }
        }
    })
}

/// The tool-args envelope with a concrete tool's parameter schema spliced in.
///
/// Constraining decoding to the *actual* tool's parameters, rather than a
/// generic object, is what makes a malformed tool call impossible instead of
/// merely unlikely.
pub fn tool_args_schema_for(tool_name: &str, parameters: &Value) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["tool", "arguments"],
        "properties": {
            "tool": { "type": "string", "const": tool_name },
            "arguments": parameters
        }
    })
}

fn route_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["entry_points"],
        "properties": {
            "entry_points": {
                "type": "array",
                "maxItems": 8,
                "items": { "type": "string", "minLength": 1, "maxLength": 128 }
            },
            "suggested_mood": {
                "type": "string",
                "enum": ["builder", "researcher", "companion"]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_structured_step_has_a_schema() {
        for s in StepKind::ALL {
            assert_eq!(
                for_step(s).is_some(),
                s.is_structured(),
                "{s} schema presence disagrees with is_structured()"
            );
        }
    }

    #[test]
    fn every_schema_forbids_extra_properties() {
        // Otherwise a model can append keys the harness silently drops, which
        // is the free-text bookkeeping the spec rules out.
        for s in StepKind::ALL {
            let Some(schema) = for_step(s) else { continue };
            assert_eq!(
                schema["additionalProperties"],
                json!(false),
                "{s} allows extra properties"
            );
        }
    }

    #[test]
    fn the_extract_schema_pins_relations_to_a_closed_set() {
        let schema = extract_schema();
        let rels = &schema["properties"]["facts"]["items"]["properties"]["relation"]["enum"];
        assert!(rels.is_array());
        assert!(rels.as_array().unwrap().contains(&json!("depends_on")));
        assert!(!rels.as_array().unwrap().contains(&json!("whatever")));
    }

    #[test]
    fn the_extract_schema_bounds_confidence() {
        let schema = extract_schema();
        let c = &schema["properties"]["facts"]["items"]["properties"]["confidence"];
        assert_eq!(c["minimum"], json!(0.0));
        assert_eq!(c["maximum"], json!(1.0));
    }

    #[test]
    fn the_extract_schema_caps_the_batch_size() {
        assert_eq!(extract_schema()["properties"]["facts"]["maxItems"], json!(16));
    }

    #[test]
    fn extraction_terms_cannot_exceed_the_verify_limit() {
        // The schema's maxLength must not be looser than the verify stage's
        // max_term_len, or the model spends tokens on facts guaranteed to be
        // rejected as malformed.
        let schema = extract_schema();
        let props = &schema["properties"]["facts"]["items"]["properties"];
        assert_eq!(props["subject"]["maxLength"], json!(128));
        assert_eq!(props["object"]["maxLength"], json!(128));
    }

    #[test]
    fn splicing_a_tool_schema_pins_the_tool_name() {
        let params = json!({
            "type": "object",
            "required": ["path"],
            "properties": { "path": { "type": "string" } }
        });
        let schema = tool_args_schema_for("read", &params);
        assert_eq!(schema["properties"]["tool"]["const"], json!("read"));
        assert_eq!(schema["properties"]["arguments"], params);
    }

    #[test]
    fn the_route_schema_only_admits_real_moods() {
        let moods = &route_schema()["properties"]["suggested_mood"]["enum"];
        assert_eq!(moods, &json!(["builder", "researcher", "companion"]));
    }
}
