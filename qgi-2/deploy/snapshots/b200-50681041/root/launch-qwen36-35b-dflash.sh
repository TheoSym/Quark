#!/usr/bin/env bash
# Qwen3.6-35B-A3B (FP8) + z-lab DFlash draft on ONE B200, sharing card 3 with the
# 27B DFlash2 worker (launch order: 27B first — SGLang's --mem-fraction-static is a
# share of the memory FREE when this process starts; 0.55 of ~90 GB ≈ 50 GB, which
# leaves ~40 GB for the DiffusionGemma NVFP4 vLLM tenant).
# SGLang main (/root/sglang-main, /venv/sglang-main). 2026-09-13.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B-FP8}"
DRAFT_PATH="${DRAFT_PATH:-/models/Qwen3.6-35B-A3B-DFlash}"
PORT="${PORT:-18036}"
NAME="${NAME:-QGI-Flash-3.6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH:-131072}" \
  --mem-fraction-static "${MEM_FRAC:-0.55}" \
  --max-running-requests "${MAX_RUNNING:-16}" \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_PATH" \
  --speculative-num-draft-tokens "${DRAFT_TOKENS:-8}" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --page-size 64 \
  --enable-metrics --enable-cache-report \
  "$@"
