#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVIDIA NVFP4 ModelOpt MIXED_PRECISION export) on ONE B200,
# SGLang main (worktree /root/sglang-main, venv /venv/sglang-main).
#
# Shape follows the SGLang cookbook's verified B200 NVFP4 low-latency cell
# (TP1, FlashInfer GDN kernels, bf16 SSM state, in-checkpoint MTP head NEXTN
# 3/1/4) plus the notes for the NVIDIA export (no --quantization: resolves to
# modelopt_mixed; --moe-runner-backend flashinfer_cutlass explicitly).
#
# KV cache: fp8_e4m3. The QSA backend reads an fp8 pool and dequantizes on
# gather (no per-tensor scales), so the pool holds 2x the tokens of bf16.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-NVFP4}"
PORT="${PORT:-18035}"
NAME="${NAME:-Qwen3.8-Flash-Next-NVFP4}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SPEC_FLAGS=()
if [ "${MTP:-1}" = "1" ]; then
  SPEC_FLAGS=(--speculative-algorithm NEXTN --speculative-num-steps 3
              --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi

# EXPORT=nvidia (default): nvidia/Qwen3.8-Flash-Next-NVFP4, ModelOpt
#   MIXED_PRECISION. Needs the flashinfer_cutlass MoE runner pinned, and the
#   47.7 GiB FP8 N-gram (PLE) table goes to pinned host RAM (the SM100
#   default) -- the loader refuses to auto-switch an offloaded table to fp8,
#   so its dtype is declared via a config override.
# EXPORT=radixark: RadixArk/Qwen3.8-Flash-Next-NVFP4, SGLang's own W4A4 build
#   (routed experts NVFP4, everything else BF16). Runs the cookbook's verified
#   B200 cell as-is: no MoE-runner pin, no PLE override.
EXPORT_FLAGS=()
if [ "${EXPORT:-nvidia}" = "nvidia" ]; then
  EXPORT_FLAGS=(--json-model-override-args '{"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}'
                --moe-runner-backend flashinfer_cutlass)
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tp 1 \
  "${EXPORT_FLAGS[@]}" \
  --context-length "${CONTEXT_LENGTH:-262144}" \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  ${MAX_RUNNING:+--max-running-requests "$MAX_RUNNING"} \
  ${MAX_MAMBA:+--max-mamba-cache-size "$MAX_MAMBA"} \
  \
  --moe-runner-backend flashinfer_cutlass \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  \
  "${SPEC_FLAGS[@]}" \
  \
  --reasoning-parser "${REASONING_PARSER:-auto}" \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  \
  --enable-metrics \
  --enable-cache-report \
  "$@"
