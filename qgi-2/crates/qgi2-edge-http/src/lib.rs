//! The OpenAI-compatible HTTP edge.
//!
//! This is the edge that needs **zero** changes to jcode. A stock jcode binary
//! reaches QGI-2 through its existing named-provider config:
//!
//! ```toml
//! # ~/.jcode/config.toml
//! [providers.qgi2]
//! type = "openai-compatible"
//! base_url = "http://127.0.0.1:8788/v1"
//! default_model = "qgi2/builder-traceable"
//! ```
//!
//! jcode sends a normal chat completion; QGI-2 runs the whole per-turn loop
//! behind it — cache-shaped assembly, planner/worker routing, constrained
//! extraction, rule validation, graph commit — and returns a normal chat
//! completion. The `usage.prompt_tokens_details.cached_tokens` in the reply is
//! the real number from vLLM, so jcode's own cache-cost UI reports QGI-2's
//! prefix-cache behaviour without knowing QGI-2 exists.
//!
//! The model name carries the persona: `qgi2/<mood>-<profile>`. That is how a
//! client with no QGI-2-specific fields still gets to pick one — and it means
//! jcode's `/model` switcher doubles as a mood switcher.

pub mod model_name;
pub mod openai;
pub mod routes;
pub mod state;

pub use model_name::{ModelName, parse_model_name};
pub use openai::{Message, ToolDeclaration, read_transcript};
pub use routes::router;
pub use state::{AppState, SessionStore};
