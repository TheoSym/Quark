#!/usr/bin/env bash
# QGI-2 single-model process: DeepSeek-V4.1-Flash on SGLang, DSpark, 4 GPUs.
#
# One process serves every step (plan, answer, route, tool-args, extract);
# qgi2 detects single-model mode from the config that points both roles here.
#
# Fits four 96 GB cards only because the Engram tables (~189 GiB of the
# checkpoint) are moved to host memory; check `free -g` shows 300 GB+ before
# launching. Weights: ./fetch-weights.sh planner
#
# ENGINE: V4.1-Flash support has not shipped in an SGLang release. Run inside
# the preview image (lmsysorg/sglang:dev-dsv41) or a build from its branch. RTX
# PRO 6000 (SM120) is NOT on the verified list (GB300/H200/B200/B300); the
# MXFP4 MoE path is the thing to watch on first load.
#
# Follows model-serve: --host 127.0.0.1, :18xxx, exposure via tailscale serve,
# and the §7 gate: ./verify-agent-ready.sh $PORT "$NAME" before qgi2 doctor.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
PORT="${PORT:-18034}"
NAME="${NAME:-QGI-2-V41}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-4}"
EP="${EP:-$TP}"

# Block size 5 is the checkpoint default; qgi2's config must declare
# speculation = "dspark", speculation_n = 5 for this endpoint. There is no MTP
# or EAGLE path for this model -- DSpark is its own bundled draft head.
DSPARK_BLOCK="${DSPARK_BLOCK:-5}"

# The pin file fetch-weights.sh leaves beside the weights, if present, so the
# log says which revision this process is running.
[ -f "$MODEL_PATH/.qgi2-pin.json" ] && echo "pinned: $(cat "$MODEL_PATH/.qgi2-pin.json")"

export CUDA_VISIBLE_DEVICES="$GPUS"
# FlashInfer's JIT detects the target arch by enumerating every GPU on the
# host. On a host where multi-GPU CUDA init is broken (see V41-BRINGUP.md) that
# probe fails, the arch list comes back empty, and every JIT module -- attention
# and the sampler -- dies with "FlashInfer requires GPUs with sm75 or higher".
# On such a host state the arch explicitly, e.g. FLASHINFER_CUDA_ARCH_LIST=12.0a
# for RTX PRO 6000 (measured working) or 10.0a for B200. Left unset here so a
# healthy host auto-detects; a wrong value compiles kernels for the wrong card.
[ -n "${FLASHINFER_CUDA_ARCH_LIST:-}" ] && export FLASHINFER_CUDA_ARCH_LIST
[ -n "${TORCH_CUDA_ARCH_LIST:-}" ] && export TORCH_CUDA_ARCH_LIST
# Engram tables in host RAM: what makes four cards enough.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="${SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE:-1}"

HICACHE_FLAGS=()
if [ "${HICACHE:-1}" = "1" ]; then
  # The dsv4 attention backend fixes the page size at 256 tokens; qgi2's
  # [hicache] page_size must say 256 or the harness pads its prefix to the
  # wrong boundary.
  HICACHE_FLAGS=(--enable-hierarchical-cache --hicache-ratio 2
                 --hicache-io-backend kernel --hicache-write-policy write_through_selective
                 --hicache-mem-layout page_first)
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --tp "$TP" --ep-size "$EP" \
  `# 0.8 is the cookbook's 4-card setting. Two B200s need >= 0.831 just to` \
  `# hold the weights plus the DSpark draft (the engine says so at load), so` \
  `# the tighter layout runs at 0.92 and caps the KV pool instead of letting` \
  `# it absorb every free byte -- CUDA graphs and activations need the rest.` \
  --mem-fraction-static "${MEM_FRAC:-0.8}" \
  ${MAX_TOTAL_TOKENS:+--max-total-tokens "$MAX_TOTAL_TOKENS"} \
  `# (no --language-model-only: SGLang's DeepseekV4ForCausalLM rejects it at` \
  `#  this commit; the class is text-only already)` \
  \
  `# --- speculation: the checkpoint's own DSpark draft head ---` \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size "$DSPARK_BLOCK" \
  `# speculation resets this to 48 when unset; state the concurrency instead` \
  --max-running-requests "${MAX_RUNNING:-64}" \
  --cuda-graph-max-bs-decode 64 \
  \
  `# --- faster prefill on the sliding-window layers ---` \
  --enable-decoder-swa-bounded-replay \
  \
  `# --- agent readiness (model-serve §7): without these the model returns` \
  `# zero function_call items, which is what QGI-2's tool-args step reads.` \
  `# explicit rather than auto: both names exist at the dsv41 commit, and a` \
  `# named parser cannot silently resolve to the wrong dialect` \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  \
  "${HICACHE_FLAGS[@]}" \
  \
  `# --- metrics: qgi2 doctor reads cache hit rate and DSpark acceptance here ---` \
  --enable-metrics \
  `# without this SGLang reports prompt_tokens_details.cached_tokens = 0 on` \
  `# every reply -- the harness then flags a cache breach on a prefix that is` \
  `# byte-identical turn to turn (measured on the first live run)` \
  --enable-cache-report
