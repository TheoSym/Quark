#!/usr/bin/env bash
# Relaunch the six-model coding-modes matrix in parallel. Arms that already
# have a clean result are skipped by run_modes_matrix.sh, so this only runs
# what is missing (e.g. after an OpenRouter credit top-up).
#
# Requires the six memory proxies on 8801-8806 (qgi2 serve --proxy --compact
# every-turn, configs in the session scratchpad qgi2-proxy/) and the keys in
# the repo .env (OPENROUTER_API_KEY, Ex_AI_API_KEY -> EXA_API_KEY).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/results/modes-2026-09-14}"
key() { python -c "import re,sys;print([l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39)) for l in open('F:/QHG/Quark/.env',encoding='utf-8',errors='replace') if l.startswith(sys.argv[1]+'=')][0])" "$1"; }
export OPENROUTER_API_KEY="$(key OPENROUTER_API_KEY)"
export EXA_API_KEY="$(key Ex_AI_API_KEY)"
echo "keys or=${#OPENROUTER_API_KEY} exa=${#EXA_API_KEY}"
for p in 8801 8802 8803 8804 8805 8806; do curl -s -m 3 -o /dev/null "http://127.0.0.1:$p/v1/models" || echo "proxy on $p is NOT up"; done

run() { bash "$HERE/run_modes_matrix.sh" "$@" "$OUT" "${6:-}" >> "$OUT/$1.run.log" 2>&1 & }
run fable-low "anthropic/claude-fable-5.1" or-fable-low qm-fable-low "" low
run opus-max  "anthropic/claude-opus-5"    or-opus-max  qm-opus-max  "" max
run gem       "Gemini 3.8 Flash"           qgillm       qm-gem
run v4pro     "QGI-V4 Pro"                 qgillm       qm-v4pro
run q38       "QGI 3.8 Flash"              qgillm       qm-q38
run glm53     "GLM 5.3"                    qgillm       qm-glm53
wait
python "$HERE/modes_report.py" "$OUT"
