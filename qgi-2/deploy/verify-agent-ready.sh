#!/usr/bin/env bash
# model-serve §7 agent-readiness gate.
#
# MANDATORY before registering a chat model. Two bringups in Aug 2026 passed
# every smoke test and were still unusable in agents: no reasoning/tool parsers,
# so thinking and raw <function=...> XML landed in `content` and zero
# function_call items came back. HTTP 200 is not readiness.
#
# QGI-2 depends on both properties directly: its tool-args step reads
# function_call items, and its extract step needs `content` to be clean JSON.
set -uo pipefail

PORT="${1:?usage: verify-agent-ready.sh <port> <served-model-name>}"
NAME="${2:?usage: verify-agent-ready.sh <port> <served-model-name>}"
BASE="http://127.0.0.1:${PORT}"
fail=0

echo "== liveness =="
for _ in $(seq 1 1800); do
  curl -sf "$BASE/v1/models" | grep -q "$NAME" && break
  sleep 1
done
curl -sf "$BASE/v1/models" | grep -q "$NAME" || { echo "FAIL: model never appeared"; exit 1; }
echo "ok: $NAME is listed"

echo
echo "== gate 1: clean content + reasoning split =="
resp=$(curl -sf -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$NAME\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],
  \"max_tokens\": 2048
}")
content=$(printf '%s' "$resp" | python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content",""))' 2>/dev/null || echo "")
if [ "$(printf '%s' "$content" | tr -d '[:space:]')" = "OK" ]; then
  echo "ok: content is clean"
else
  echo "FAIL: content was $(printf '%s' "$content" | head -c 120)"
  echo "      -> missing --reasoning-parser; thinking is leaking into content"
  fail=1
fi

echo
echo "== gate 2: a real function_call item =="
# Through /v1/responses, per §7 — the shape agents actually consume.
resp=$(curl -sf -X POST "$BASE/v1/responses" -H 'Content-Type: application/json' -d "{
  \"model\": \"$NAME\",
  \"input\": \"list /tmp using the shell tool\",
  \"tools\": [{\"type\":\"function\",\"name\":\"shell\",\"description\":\"run a shell command\",
              \"parameters\":{\"type\":\"object\",\"properties\":{\"cmd\":{\"type\":\"string\"}},\"required\":[\"cmd\"]}}],
  \"max_output_tokens\": 2048
}" 2>/dev/null || echo '{}')
if printf '%s' "$resp" | grep -q '"function_call"'; then
  echo "ok: function_call present"
else
  echo "FAIL: no function_call item — text-only output"
  echo "      -> missing --tool-call-parser qwen3_coder (qwen25 is the JSON"
  echo "         dialect and silently fails on the <function=...> XML one)"
  fail=1
fi

echo
echo "== QGI-2 extras =="
# QGI-2 constrains every structured step, so the guided path must work or the
# extract step returns prose and the turn fails on a parse error.
resp=$(curl -sf -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$NAME\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Give me an object with a single key ok set to true.\"}],
  \"max_tokens\": 512,
  \"response_format\": {\"type\":\"json_schema\",\"json_schema\":{\"name\":\"probe\",\"strict\":true,
    \"schema\":{\"type\":\"object\",\"additionalProperties\":false,\"required\":[\"ok\"],
               \"properties\":{\"ok\":{\"type\":\"boolean\"}}}}}
}")
body=$(printf '%s' "$resp" | python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content",""))' 2>/dev/null || echo "")
if printf '%s' "$body" | python -c 'import json,sys; d=json.load(sys.stdin); assert isinstance(d.get("ok"), bool)' 2>/dev/null; then
  echo "ok: guided JSON decoding works"
else
  echo "FAIL: constrained decode did not return the schema; got $(printf '%s' "$body" | head -c 120)"
  echo "      -> every QGI-2 structured step would fail on a parse error"
  fail=1
fi

# Prefix caching is the whole premise; if cached_tokens never appears the
# harness cannot measure the metric the spec calls a bug when it drops.
if curl -sf "$BASE/metrics" 2>/dev/null | grep -q 'sglang:cache_hit_rate\|sglang:spec_accept_length'; then
  echo "ok: /metrics exposes cache and acceptance gauges"
else
  echo "WARN: no cache/acceptance gauges — add --enable-metrics"
fi

echo
[ "$fail" -eq 0 ] && echo "AGENT-READY" || { echo "NOT AGENT-READY ($fail gate failure(s))"; exit 1; }
