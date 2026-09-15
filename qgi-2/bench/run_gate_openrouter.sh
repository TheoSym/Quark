#!/usr/bin/env bash
# The gate over frontier models on OpenRouter, direct (no gateway), with a
# pinned reasoning effort per model. Sequential, one JSON + one log per run.
# Cost per leg comes from OpenRouter's per-token pricing (jcode legs) and
# from usage.cost where the API returns it (direct legs).
#
# Usage: run_gate_openrouter.sh            # the list below
#        run_gate_openrouter.sh "profile model effort" ...
# Needs OPENROUTER_API_KEY in the environment or the repo .env (gate.py
# lifts it into jcode's environment; the value is never printed).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-/f/QHG/polyglot-benchmark}"
LEGS="${LEGS:-engine coding agentic memory}"
OUT="$HERE/results/gate"; mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8

# profile  model  effort ("-" = none)
if [ $# -gt 0 ]; then RUNS=("$@"); else RUNS=(
  "or-fable-low        anthropic/claude-fable-5.1        low"
  "or-opus-max         anthropic/claude-opus-5           max"
  "or-fable-batch-low  anthropic/claude-fable-5.1:batch  low"
  "or-mercury          inception/mercury-2.5             -"
  "or-grok             x-ai/grok-4.6                     -"
  "or-astra-high       openai/gpt-6-astra                high"
); fi

for spec in "${RUNS[@]}"; do
  set -- $spec; prof=$1; model=$2; effort=$3
  tag="$(echo "$model-$effort" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
  if [ -f "$OUT/gate-$tag.json" ]; then echo "== skip $model ($effort): gate-$tag.json exists"; continue; fi
  echo "== $model  effort=$effort  $(date +%H:%M)"
  extra=(); [ "$effort" != "-" ] && extra=(--reasoning-effort "$effort")
  python "$HERE/gate.py" --model "$model" --jcode-profile "$prof" --bench "$BENCH" --legs $LEGS \
    --tag "$tag" "${extra[@]}" > "$OUT/run-$tag.log" 2>&1
  tail -2 "$OUT/run-$tag.log"
done
echo "== report"
python "$HERE/gate.py" --report "$OUT"
