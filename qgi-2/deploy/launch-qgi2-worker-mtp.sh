#!/usr/bin/env bash
#
# OPTIONAL. This is a second worker for throughput (c8+), not a profile
# requirement: the DFlash2 worker already serves `deterministic` because
# DFlash2 runs greedy (measured). See deploy/README.md.
# QGI-2 worker (MTP) — the second worker, for the `deterministic` profile only.
#
# Why this exists: `traceable` runs the worker on DFlash2 n=7, but DFlash2
# cannot produce greedy output, and `deterministic` forces T=0. One process
# cannot serve both profiles, so the spec's "two processes" is a floor.
#
# Skip this entirely if you only run `traceable`.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the worker checkpoint}"
PORT="${PORT:-18034}"
NAME="${NAME:-QGI-2-Worker-MTP}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# Must match qgi2's declared worker speculation for the deterministic profile.
SPEC_STEPS="${SPEC_STEPS:-3}"

export CUDA_VISIBLE_DEVICES="$GPU"

# On Ampere (A6000, sm_86) an FP8 checkpoint fails on the first forward pass.
# Point MODEL_PATH at an AWQ/GPTQ int4 build there; on Blackwell FP8 is fine.
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length 131072 \
  --mem-fraction-static 0.80 \
  \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  \
  --speculative-algorithm NEXTN \
  --speculative-num-steps "$SPEC_STEPS" \
  --speculative-num-draft-tokens "$((SPEC_STEPS + 1))" \
  \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through_selective \
  --hicache-mem-layout page_first \
  \
  --enable-metrics
