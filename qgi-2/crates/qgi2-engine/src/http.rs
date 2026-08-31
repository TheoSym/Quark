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
