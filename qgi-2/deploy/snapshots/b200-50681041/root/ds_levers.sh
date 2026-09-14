#!/bin/bash
# Sweep DeepSeek-V4.1-Flash speed levers on the planner, one restart each,
# same probe each time; restores the original wrapper at the end.
set -u
W=/opt/supervisor-scripts/qgi2-v41.sh
cp "$W" "$W.pre-levers"
LINE='export PORT=18034 NAME=QGI-2-V41 CUDA_VISIBLE_DEVICES="0,1" TP="2" HICACHE=1 MAX_RUNNING=32 MEM_FRAC=0.92 MAX_TOTAL_TOKENS=4000000'
restart_and_probe() {  # $1 label, $2 extra env (space separated KEY=VAL), $3 extra launch flags
  sed -i "s|^export PORT=18034 .*|$LINE $2|" "$W"
  sed -i "s|^pty ./launch-qgi2-planner-v41.sh.*|pty ./launch-qgi2-planner-v41.sh $3 2>\&1|" "$W"
  supervisorctl restart qgi2-v41 >/dev/null
  for i in $(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:18034/health && break; sleep 10; done
  if ! curl -s -m 3 -o /dev/null http://127.0.0.1:18034/health; then echo "== $1: FAILED TO START"; tail -c 3000 /var/log/portal/qgi2-v41.log | grep -i "error\|Traceback" | tail -5; return; fi
  echo "== $1 (env: ${2:-none}; flags: ${3:-none})"
  python3 /root/ds_speed.py
}
restart_and_probe "A: baseline" "" ""
restart_and_probe "B: Q8KV8 sparse prefill" "SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH=1" "--dsv4-prefill-backend flashmla_sparse_q8"
restart_and_probe "C: DeepEP all-to-all (EP2)" "" "--moe-a2a-backend deepep"
restart_and_probe "D: DSpark fused greedy markov" "SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV=1" ""
cp "$W.pre-levers" "$W"
supervisorctl restart qgi2-v41 >/dev/null
for i in $(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:18034/health && break; sleep 10; done
echo "== restored original; planner $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18034/health)"
