//! The shared HTTP client.
//!
//! One `reqwest::Client` is shared across every endpoint and both backends so
//! connection pooling and HTTP/2 multiplexing work. The planner and worker are
//! typically two processes on the same host, and a per-endpoint client would
//! open a fresh connection for every step of every turn.

use crate::endpoint::Endpoint;
use anyhow::{Context, Result, bail};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: reqwest::Client,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new(Duration::from_secs(600))
    }
}

impl HttpClient {
    pub fn new(timeout: Duration) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(timeout)
            // Long generations under speculation can be quiet between chunks; a
            // short idle timeout would kill them mid-stream.
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("reqwest client with default TLS should build");
        Self { inner }
    }

    pub fn raw(&self) -> &reqwest::Client {
        &self.inner
    }

    /// POST JSON to an endpoint path and return the body text.
    ///
    /// A non-2xx response carries the server's own complaint — a schema the
    /// guided backend rejected, a model name that is not served. Dropping it
    /// for a bare status code makes those undiagnosable, so the body is folded
    /// into the error.
    pub async fn post_json<B: serde::Serialize>(
        &self,
        endpoint: &Endpoint,
        path: &str,
        body: &B,
    ) -> Result<String> {
        let url = endpoint.url(path);
        let mut req = self.inner.post(&url).json(body);
        if let Some(key) = &endpoint.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let text = resp.text().await.context("reading response body")?;
        if !status.is_success() {
            bail!(
                "{} returned {status} from {}: {text}",
                endpoint.engine,
                endpoint.base_url
            );
        }
        Ok(text)
    }

    /// POST JSON and collect a Server-Sent Events stream into one body.
    ///
    /// Streaming is the default for QGI-2 because a non-streamed request that
    /// runs past ~100 s is killed by the Cloudflare edge with a 524, and a
    /// planner step on a long prompt does exactly that. Streaming also keeps
    /// the connection alive for the whole generation, so the failure mode
    /// becomes a slow answer rather than a lost turn.
    ///
    /// Returns `(text, usage)`. Usage arrives on the final chunk and only when
    /// `stream_options.include_usage` was set — without it the cache-hit metric
    /// silently reads zero for every streamed turn.
    pub async fn post_sse<B: serde::Serialize>(
        &self,
        endpoint: &Endpoint,
        path: &str,
        body: &B,
    ) -> Result<(String, Option<serde_json::Value>)> {
        use futures::StreamExt;

        let url = endpoint.url(path);
        let mut req = self.inner.post(&url).json(body);
        if let Some(key) = &endpoint.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.with_context(|| format!("POST {url} (stream)"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!(
                "{} returned {status} from {}: {text}",
                endpoint.engine,
                endpoint.base_url
            );
        }

        let mut text = String::new();
        let mut usage = None;
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("reading SSE chunk")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // Events are separated by a blank line, and a network chunk can
            // split one, so only complete events are consumed.
            while let Some(idx) = buf.find("

") {
                let event = buf[..idx].to_string();
                buf.drain(..idx + 2);
                collect_event(&event, &mut text, &mut usage);
            }
        }
        Ok((text, usage))
    }

    /// GET a URL and return the body, for Prometheus scrapes.
    pub async fn get_text(&self, url: &str, api_key: Option<&str>) -> Result<String> {
        let mut req = self.inner.get(url).timeout(Duration::from_secs(10));
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.with_context(|| format!("GET {url}"))?;
        resp.text().await.context("reading body")
    }

    /// Whether an endpoint answers a model listing.
    ///
    /// A 2xx is not enough. A gateway behind Cloudflare Access answers an
    /// unauthenticated request with an HTML login page and status 200, so a
    /// status-only check reports a reachable-but-unusable endpoint as healthy —
    /// and the failure then surfaces on the first turn as unparseable JSON. The
    /// body has to look like a model listing.
    pub async fn health(&self, endpoint: &Endpoint) -> bool {
        self.health_detail(endpoint).await.is_ok()
    }

    /// Health with a reason, for `qgi2 doctor`.
    pub async fn health_detail(&self, endpoint: &Endpoint) -> Result<(), String> {
        let url = format!("{}/models", endpoint.base_url.trim_end_matches('/'));
        let mut req = self.inner.get(&url).timeout(Duration::from_secs(5));
        if let Some(key) = &endpoint.api_key {
            req = req.bearer_auth(key);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return Err(format!("unreachable: {e}")),
        };
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format!("HTTP {status}"));
        }
        let looks_like_html = body.trim_start().starts_with('<');
        if looks_like_html {
            return Err(
                "answered HTML, not JSON — likely a Cloudflare Access login page. \
                 Send CF-Access-Client-Id/Secret, or call over the tailnet."
                    .into(),
            );
        }
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) if v.get("data").is_some() || v.get("object").is_some() => Ok(()),
            Ok(_) => Err("answered JSON that is not a model listing".into()),
            Err(_) => Err("answered something that is not JSON".into()),
        }
    }
}

/// Fold one SSE event into the accumulating text and usage.
fn collect_event(event: &str, text: &mut String, usage: &mut Option<serde_json::Value>) {
    for line in event.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        // A chunk shape this client does not know is not worth failing a turn
        // over; the deltas and the usage chunk are what matter.
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if let Some(u) = v.get("usage")
            && !u.is_null()
        {
            *usage = Some(u.clone());
        }
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            text.push_str(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qgi2_spec_types::Speculation;

    #[test]
    fn a_client_builds_with_defaults() {
        let _ = HttpClient::default();
    }

    #[test]
    fn paths_join_onto_the_endpoint_base() {
        let e = Endpoint::new("http://127.0.0.1:30000/v1", "m", Speculation::Off);
        assert_eq!(e.url("/chat/completions"), "http://127.0.0.1:30000/v1/chat/completions");
        assert_eq!(e.url("/embeddings"), "http://127.0.0.1:30000/v1/embeddings");
    }
}

#[cfg(test)]
mod health_tests {
    use super::*;

    // The body checks are pure, so they are testable without a server. The bug
    // they pin: a Cloudflare Access login page is HTTP 200, so a status-only
    // health check called an unusable gateway "up" and the failure surfaced on
    // the first turn as unparseable JSON instead.
    fn classify(body: &str) -> Result<(), String> {
        if body.trim_start().starts_with('<') {
            return Err("html".into());
        }
        match serde_json::from_str::<serde_json::Value>(body) {
            Ok(v) if v.get("data").is_some() || v.get("object").is_some() => Ok(()),
            Ok(_) => Err("not a listing".into()),
            Err(_) => Err("not json".into()),
        }
    }

    #[test]
    fn a_real_model_listing_is_healthy() {
        assert!(classify(r#"{"object":"list","data":[{"id":"m"}]}"#).is_ok());
    }

    #[test]
    fn a_cloudflare_login_page_is_not_healthy_despite_being_200() {
        assert!(classify("<!DOCTYPE html><html><body>Sign in</body></html>").is_err());
        assert!(classify("\n  <html>").is_err());
    }

    #[test]
    fn json_that_is_not_a_listing_is_not_healthy() {
        assert!(classify(r#"{"error":"nope"}"#).is_err());
    }

    #[test]
    fn an_empty_body_is_not_healthy() {
        assert!(classify("").is_err());
    }
}

#[cfg(test)]
mod sse_tests {
    use super::*;

    fn collect(events: &[&str]) -> (String, Option<serde_json::Value>) {
        let mut text = String::new();
        let mut usage = None;
        for e in events {
            collect_event(e, &mut text, &mut usage);
        }
        (text, usage)
    }

    #[test]
    fn deltas_accumulate_into_the_answer() {
        let (text, _) = collect(&[
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":", world"}}]}"#,
        ]);
        assert_eq!(text, "Hello, world");
    }

    #[test]
    fn the_final_usage_chunk_is_captured() {
        // Without stream_options.include_usage there is no such chunk, and the
        // cache-hit metric reads zero for every streamed turn.
        let (_, usage) = collect(&[
            r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":90}}}"#,
        ]);
        let u = usage.expect("usage chunk");
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 90);
    }

    #[test]
    fn done_and_blank_lines_are_ignored() {
        let (text, _) = collect(&[
            r#"data: {"choices":[{"delta":{"content":"x"}}]}"#,
            "data: [DONE]",
            "data:",
            ": a comment",
        ]);
        assert_eq!(text, "x");
    }

    #[test]
    fn an_unparseable_chunk_does_not_lose_the_turn() {
        // A chunk shape this client does not know should be skipped, not fatal.
        let (text, _) = collect(&[
            r#"data: {"choices":[{"delta":{"content":"a"}}]}"#,
            "data: {not json",
            r#"data: {"choices":[{"delta":{"content":"b"}}]}"#,
        ]);
        assert_eq!(text, "ab");
    }

    #[test]
    fn a_null_usage_field_is_not_mistaken_for_usage() {
        // vLLM sends "usage": null on every non-final chunk.
        let (_, usage) = collect(&[r#"data: {"choices":[{"delta":{"content":"x"}}],"usage":null}"#]);
        assert!(usage.is_none());
    }
}
