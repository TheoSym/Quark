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
    pub async fn health(&self, endpoint: &Endpoint) -> bool {
        let url = format!("{}/models", endpoint.base_url.trim_end_matches('/'));
        let mut req = self.inner.get(url).timeout(Duration::from_secs(5));
        if let Some(key) = &endpoint.api_key {
            req = req.bearer_auth(key);
        }
        matches!(req.send().await, Ok(r) if r.status().is_success())
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
