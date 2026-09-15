#!/usr/bin/env bash
# Coding-modes matrix for one model: six arms, three exercises each.
#
#   baseline          jcode, its own prompt
#   safe-efficient    + prompts/coding-modes/safe-efficient-coding.system.md
#   exa-reuse         + prompts/coding-modes/exa-search-reuse.system.md
#   exa-reuse+tool    + the same overlay + the Exa MCP server (needs EXA_API_KEY)
#   mem               jcode through the QGI-2 memory proxy, baseline prompt
#   mem+exa-tool      memory proxy + exa-reuse overlay + Exa MCP server
#
# Usage: run_modes_matrix.sh <key> <served-model> <direct-profile> <proxy-profile> <out_dir> [effort]
#   e.g. run_modes_matrix.sh fable-low anthropic/claude-fable-5.1 or-fable-low qm-fable-low results/modes-2026-09-14 low
# The proxy for <proxy-profile> must already be up (qgi2 serve --proxy --compact every-turn).
# Tool arms are written as BLOCKED, not run, when EXA_API_KEY is absent: a
# tool arm without the tool would silently duplicate the overlay-only arm.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$1"; MODEL="$2"; DPROF="$3"; QPROF="$4"; OUT="$5"; EFFORT="${6:-}"
BENCH="${BENCH:-/f/QHG/polyglot-benchmark}"
JCODE="${JCODE:-$HERE/../../target/selfdev/jcode.exe}"
EXERCISES="${EXERCISES:-bowling forth wordy}"
TIMEOUT="${TIMEOUT:-600}"
MODES="$HERE/../prompts/coding-modes"
SAFE="$MODES/safe-efficient-coding.system.md"
EXA="$MODES/exa-search-reuse.system.md"
MCP="$MODES/exa.mcp.json"
mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

run() {  # label profile [extra polyglot args...]
  local label="$1" prof="$2"; shift 2
  local out="$OUT/$KEY-$label.json"
  if [ -f "$out" ]; then echo "  skip $label (exists)"; return; fi
  echo "  == $KEY $label  $(date +%H:%M:%S)"
  python "$HERE/polyglot_bench.py" --bench "$BENCH" --lang python --exercises $EXERCISES \
    --agents jcode --jcode "$JCODE" --jcode-profile "$prof" --model "$MODEL" \
    --timeout "$TIMEOUT" --label "$label" --out "$out" "$@" > "$OUT/$KEY-$label.log" 2>&1
  python - "$out" <<'PY' 2>/dev/null || tail -3 "$OUT/$KEY-$label.log"
import json,sys
s=list(json.load(open(sys.argv[1],encoding='utf-8'))['summary'].values())[0]
print(f"     {s['solved']}/{s['n']} med={s['median_wall_s']}s tot={s['total_wall_s']}s in={s['input_tokens']} out={s['output_tokens']} err={s['errors']}")
PY
}
blocked() {  # label reason
  local out="$OUT/$KEY-$1.json"
  [ -f "$out" ] && return
  python - "$out" "$MODEL" "$1" "$2" <<'PY'
import json,sys
json.dump({"model":sys.argv[2],"lang":"python","exercises":[],"summary":{sys.argv[3]:{"n":0,"solved":0,"status":"BLOCKED"}},
           "results":[],"note":sys.argv[4]},open(sys.argv[1],"w",encoding="utf-8"),indent=1)
PY
  echo "  == $KEY $1: BLOCKED ($2)"
}

run baseline        "$DPROF"
run safe-efficient  "$DPROF" --jcode-overlay "$SAFE"
run exa-reuse       "$DPROF" --jcode-overlay "$EXA"
if [ -n "${EXA_API_KEY:-}" ]; then
  run exa-reuse+tool "$DPROF" --jcode-overlay "$EXA" --jcode-mcp "$MCP"
else
  blocked exa-reuse+tool "EXA_API_KEY not available in this environment"
fi
run mem             "$QPROF" --session-per-task
if [ -n "${EXA_API_KEY:-}" ]; then
  run mem+exa-tool  "$QPROF" --session-per-task --jcode-overlay "$EXA" --jcode-mcp "$MCP"
else
  blocked mem+exa-tool "EXA_API_KEY not available in this environment"
fi
echo "DONE $KEY $(date +%H:%M:%S)"
