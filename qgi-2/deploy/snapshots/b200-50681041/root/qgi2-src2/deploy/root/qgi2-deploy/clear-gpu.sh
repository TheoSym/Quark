#!/usr/bin/env bash
# model-serve §0 — a crashed server pins VRAM through its worker processes, so
# the next launch OOMs on a GPU that nvidia-smi shows as busy with nothing.
set -uo pipefail

supervisorctl stop vllm 2>/dev/null || true

# The worker procs share a process group with the engine core; killing the
# group is what actually releases the memory.
if pid=$(pgrep -f 'VLLM::EngineCor' | head -1); then
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null || true
fi

pkill -9 -f 'vllm serve'      2>/dev/null || true
pkill -9 -f 'sglang.launch'   2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true

sleep 2
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
