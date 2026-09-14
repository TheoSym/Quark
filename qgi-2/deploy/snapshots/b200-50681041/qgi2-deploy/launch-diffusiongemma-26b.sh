#!/usr/bin/env bash
# DiffusionGemma-26B-A4B-it (NVIDIA ModelOpt NVFP4 export, ~14 GB) on vLLM, third tenant of
# card 3 (after the 27B DFlash2 worker and the Qwen3.6 DFlash engine). vLLM's
# --gpu-memory-utilization is a share of the WHOLE card (183 GB): 0.17 ≈ 31 GB, which must be
# free when this starts. Recipe = model-serve skill "DiffusionGemma NVFP4, TP=1":
# TRITON_ATTN backend + max_new_tokens override + V2 model runner. 2026-09-13.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/models/diffusiongemma-26B-A4B-it-NVFP4}"
PORT="${PORT:-18041}"
NAME="${NAME:-QGI-DGemma-26B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_USE_V2_MODEL_RUNNER=1

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --quantization modelopt \
  --attention-backend TRITON_ATTN \
  --override-generation-config '{"max_new_tokens": null}' \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_SEQS:-8}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.17}" \
  "$@"
