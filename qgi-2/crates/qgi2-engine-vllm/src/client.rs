//! HTTP client for a vLLM endpoint.

use crate::registry::Endpoint;
use crate::types::{ChatChunk, ChatRequest, ChatResponse, EmbeddingResponse};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde_json::json;
use std::time::Duration;

/// A client for one or more vLLM endpoints.
///
/// One `reqwest::Client` is shared across endpoints so connection pooling and
/// HTTP/2 multiplexing work: the planner and worker are typically two processes
/// on the same host, and a per-endpoint client would open a fresh connection
/// for every step of every turn.
#[derive(Debug, Clone)]
pub struct VllmClient {
    http: reqwest::Client,
}

impl Default for VllmClient {
    fn default() -> Self {
        Self::new(Duration::from_secs(600))
    }
}

impl VllmClient {
    pub fn new(timeout: Duration) -> Self {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            // Long generations under speculation can be quiet between chunks;
            // a short idle timeout would kill them mid-stream.
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("reqwest client with default TLS should build");
        Self { http }
    }

    fn request(&self, endpoint: &Endpoint, path: &str) -> reqwest::RequestBuilder {
        let mut b = self.http.post(endpoint.url(path));
        if let Some(key) = &endpoint.api_key {
            b = b.bearer_auth(key);
        }
        b
    }

    /// A non-streaming chat completion.
    pub async fn chat(&self, endpoint: &Endpoint, req: &ChatRequest) -> Result<ChatResponse> {
        let resp = self
            .request(endpoint, "/chat/completions")
            .json(req)
            .send()
            .await
            .with_context(|| format!("POST {}", endpoint.url("/chat/completions")))?;

        let status = resp.status();
        let body = resp.text().await.context("reading response body")?;
        if !status.is_success() {
            // The body carries vLLM's actual complaint — a schema the guided
            // backend rejected, a model name that is not served. Dropping it
            // for a bare status code makes those undiagnosable.
            bail!("vLLM returned {status} from {}: {body}", endpoint.base_url);
        }

        serde_json::from_str(&body)
            .with_context(|| format!("decoding chat response from {}", endpoint.base_url))
    }

    /// A streaming chat completion.
    ///
    /// Returns the accumulated text and the final usage. The usage arrives on
    /// the last chunk only when `stream_options.include_usage` is set, which
    /// [`ChatRequest::streaming`] does — without it the cache-hit metric reads
    /// zero for every streamed turn.
    pub async fn chat_stream_collect(
        &self,
        endpoint: &Endpoint,
        req: &ChatRequest,
    ) -> Result<ChatResponse> {
        let mut req = req.clone();
        if !req.stream {
            req = req.streaming();
        }

        let resp = self
            .request(endpoint, "/chat/completions")
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {} (stream)", endpoint.url("/chat/completions")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("vLLM returned {status} from {}: {body}", endpoint.base_url);
        }

        let mut text = String::new();
        let mut usage = None;
        let mut finish_reason = None;
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading SSE chunk")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // SSE events are separated by a blank line; a chunk boundary can
            // split one, so only complete events are consumed and the
            // remainder stays buffered.
            while let Some(idx) = buf.find("\n\n") {
                let event = buf[..idx].to_string();
                buf.drain(..idx + 2);

                for line in event.lines() {
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    let parsed: ChatChunk = match serde_json::from_str(data) {
                        Ok(c) => c,
                        // A chunk shape this client does not know is not worth
                        // failing a turn over; the usage chunk and the deltas
                        // are what matter.
                        Err(_) => continue,
                    };
                    if let Some(u) = parsed.usage {
                        usage = Some(u);
                    }
                    for choice in parsed.choices {
                        if let Some(c) = choice.delta.content {
                            text.push_str(&c);
                        }
                        if choice.finish_reason.is_some() {
                            finish_reason = choice.finish_reason;
                        }
                    }
                }
            }
        }

        Ok(ChatResponse {
            id: String::new(),
            model: req.model.clone(),
            choices: vec![crate::types::ChatChoice {
                index: 0,
                message: crate::types::ChatMessage::assistant(text),
                finish_reason,
            }],
            usage,
        })
    }

    /// Embed texts. Spec: entry-point retrieval only.
    pub async fn embed(&self, endpoint: &Endpoint, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let resp = self
            .request(endpoint, "/embeddings")
            .json(&json!({ "model": endpoint.model, "input": texts }))
            .send()
            .await
            .with_context(|| format!("POST {}", endpoint.url("/embeddings")))?;

        let status = resp.status();
        let body = resp.text().await.context("reading embeddings body")?;
        if !status.is_success() {
            bail!("embedder returned {status} from {}: {body}", endpoint.base_url);
        }

        let parsed: EmbeddingResponse =
            serde_json::from_str(&body).context("decoding embeddings response")?;

        // Sort by index: the API does not guarantee response order matches
        // input order, and a mismatch would silently attach every embedding to
        // the wrong node.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        if data.len() != texts.len() {
            bail!(
                "embedder returned {} vectors for {} inputs",
                data.len(),
                texts.len()
            );
        }
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    /// Whether an endpoint is up, for the startup preflight.
    pub async fn health(&self, endpoint: &Endpoint) -> bool {
        let url = format!(
            "{}/models",
            endpoint.base_url.trim_end_matches('/')
        );
        let mut b = self.http.get(url).timeout(Duration::from_secs(5));
        if let Some(key) = &endpoint.api_key {
            b = b.bearer_auth(key);
        }
        matches!(b.send().await, Ok(r) if r.status().is_success())
    }

    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use qgi2_spec_types::Speculation;

    #[test]
    fn a_client_builds_with_defaults() {
        let _ = VllmClient::default();
    }

    #[test]
    fn requests_target_the_endpoints_chat_path() {
        let e = Endpoint::new("http://127.0.0.1:8000/v1", "m", Speculation::Off);
        assert_eq!(e.url("/chat/completions"), "http://127.0.0.1:8000/v1/chat/completions");
        assert_eq!(e.url("/embeddings"), "http://127.0.0.1:8000/v1/embeddings");
    }

    #[tokio::test]
    async fn embedding_an_empty_list_makes_no_request() {
        let c = VllmClient::default();
        let e = Endpoint::new("http://127.0.0.1:1/v1", "m", Speculation::Off);
        assert!(c.embed(&e, &[]).await.unwrap().is_empty());
    }

    #[test]
    fn a_streaming_request_is_forced_to_include_usage() {
        let req = ChatRequest::new("m", vec![ChatMessage::user("hi")]);
        assert!(req.stream_options.is_none());
        let streaming = req.streaming();
        assert!(streaming.stream_options.unwrap().include_usage);
    }
}
