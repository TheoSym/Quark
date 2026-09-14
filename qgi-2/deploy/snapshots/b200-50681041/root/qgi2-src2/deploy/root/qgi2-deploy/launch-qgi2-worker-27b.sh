#!/usr/bin/env bash
# QGI-2 27B worker on ONE B200 (shares the card with its sibling below).
# SGLang main (/root/sglang-main worktree, /venv/sglang-main).
#
# Tenants of card 3, selected by ROLE:
#   ROLE=traceable  Qwen/Qwen3.8-27B-FP8 + incoai/Qwen3.8-27B-DFlash2 (DFLASH, block 8).
#   ROLE=muse       Meta Muse-Glimmer-30B as RadixArk NVFP4 + Meta's DFlash draft.
# (The `deterministic` role -- RadixArk 27B NVFP4 + DSpark -- was removed on
#  2026-09-13 when that profile was cancelled.)
#
# Both: fp8_e4m3 KV (the cookbook pins it for every 27B recipe), HiCache with
# 64-token pages, metrics + cache report for qgi2 doctor. SGLang checks
# --mem-fraction-static against the memory FREE when that process starts, so a
# later tenant on the shared card needs a HIGHER fraction; launch traceable
# first, muse second.
set -euo pipefail

ROLE="${ROLE:?set ROLE=traceable|deterministic}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PARSERS=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
case "$ROLE" in
  traceable)
    MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B-FP8}"
    DRAFT_PATH="${DRAFT_PATH:-/models/Qwen3.8-27B-DFlash2}"
    PORT="${PORT:-18037}"; NAME="${NAME:-QGI-2-Worker-DFlash2}"
    MEM_FRAC="${MEM_FRAC:-0.45}"
    SPEC_FLAGS=(--speculative-algorithm DFLASH
                --speculative-draft-model-path "$DRAFT_PATH"
                --speculative-num-draft-tokens 8)
    ;;
  muse)
    # Third tenant of card 3: Meta's Muse-Glimmer-30B (dense, text-only in
    # this export) as RadixArk NVFP4 + MXFP8, with Meta's native DFlash draft.
    # Vision tower off (--language-model-only) -- the NVFP4 export is text only.
    MODEL_PATH="${MODEL_PATH:-/models/Muse-Glimmer-NVFP4}"
    DRAFT_PATH="${DRAFT_PATH:-/models/Muse-Glimmer-30B-assistant}"
    PORT="${PORT:-18039}"; NAME="${NAME:-Muse-Glimmer-30B-NVFP4}"
    MEM_FRAC="${MEM_FRAC:-0.20}"
    SPEC_FLAGS=(--speculative-algorithm DFLASH
                --speculative-draft-model-path "$DRAFT_PATH"
                --language-model-only)
    PARSERS=(--reasoning-parser muse --tool-call-parser muse)
    ;;
  *) echo "unknown ROLE=$ROLE" >&2; exit 2 ;;
esac

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH:-131072}" \
  --mem-fraction-static "$MEM_FRAC" \
  ${MAX_RUNNING:+--max-running-requests "$MAX_RUNNING"} \
  ${MAX_TOTAL_TOKENS:+--max-total-tokens "$MAX_TOTAL_TOKENS"} \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype "${SSM_DTYPE:-bfloat16}" \
  "${SPEC_FLAGS[@]}" \
  "${PARSERS[@]}" \
  --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-io-backend kernel --hicache-write-policy write_through_selective \
  --hicache-mem-layout page_first \
  --enable-metrics --enable-cache-report \
  "$@"
