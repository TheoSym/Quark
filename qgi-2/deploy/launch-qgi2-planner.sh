#!/usr/bin/env bash
# QGI-2 planner — SGLang, MTP speculation, HiCache L2.
#
# SGLang rather than vLLM for both QGI-2 processes: HiCache is SGLang's, and one
# engine across the pair means one cache configuration to reason about. The
# harness supports either.
#
# Follows model-serve: --host 127.0.0.1, :18xxx, exposure via tailscale serve.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the planner checkpoint}"
PORT="${PORT:-18033}"
NAME="${NAME:-QGI-2-Planner}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# Speculation MUST match what qgi2's config declares for the planner. The
# harness routes each step to the process launched with its speculation, so a
# mismatch here makes acceptance numbers describe a configuration nobody chose.
# n=2 is the spec's planner setting.
SPEC_STEPS="${SPEC_STEPS:-2}"

export CUDA_VISIBLE_DEVICES="$GPU"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  \
  `# --- agent readiness (model-serve §7). Without these the model dumps raw` \
  `# <function=...> XML into content and returns zero function_call items,` \
  `# which is precisely what QGI-2's tool-args step reads.` \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  \
  `# --- speculation: NEXTN is SGLang's name for a model's own MTP head ---` \
  --speculative-algorithm NEXTN \
  --speculative-num-steps "$SPEC_STEPS" \
  --speculative-num-draft-tokens "$((SPEC_STEPS + 1))" \
  \
  `# --- HiCache: L1 GPU -> L2 host. page-size must match qgi2's [hicache]` \
  `# page_size, or the harness pads its prefix to the wrong boundary.` \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through_selective \
  --hicache-mem-layout page_first \
  \
  `# --- metrics: qgi2 doctor and the cache/acceptance numbers read /metrics ---` \
  --enable-metrics
