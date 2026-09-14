#!/usr/bin/env bash
# Run the gate over a list of gateway (LiteLLM) models, sequentially, one
# JSON + one log per model under results/gate/. Sequential on purpose: the
# gateway is shared, and speed numbers only mean something at concurrency 1.
#
# Usage: run_gate_gateway.sh [model ...]
#   no args = the default shortlist below. Set PROFILE to use a different
#   [providers.X] block, BENCH for another polyglot checkout, LEGS to limit legs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-qgillm}"
BENCH="${BENCH:-/f/QHG/polyglot-benchmark}"
LEGS="${LEGS:-engine coding agentic memory}"
OUT="$HERE/results/gate"; mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8

if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(
  "Gemini 3.8 Flash"
  "Deepseek V4 Pro"
  "QGI 3.8 Flash"
  "GLM 5.3"
  "GLM 5.3 Flash"
  "Kimi-k3"
  "GPT-5.6 Luna"
  "Qwen 3.8 max"
  "QGI-V4 Pro"
  "QGI-Flash 3.8 Next"
); fi

for m in "${MODELS[@]}"; do
  tag="$(echo "$m" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
  if [ -f "$OUT/gate-$tag.json" ]; then echo "== skip $m (gate-$tag.json exists)"; continue; fi
  echo "== $m  $(date +%H:%M)"
  python "$HERE/gate.py" --model "$m" --jcode-profile "$PROFILE" --bench "$BENCH" --legs $LEGS \
    > "$OUT/run-$tag.log" 2>&1
  tail -2 "$OUT/run-$tag.log"
done
echo "== report"
python "$HERE/gate.py" --report "$OUT"
