set -u
up() { for i in $(seq 1 $2); do curl -sf -m 3 http://127.0.0.1:$1/v1/models >/dev/null && { echo "$1 UP after $((i*10))s"; return 0; }; sleep 10; done; echo "$1 NOT UP"; return 1; }
supervisorctl stop qgi2-dgemma qgi2-qwen36 qgi2-w27b-dflash2 >/dev/null; sleep 5
cp /root/qgi2-w27b-dflash2.sh /root/qgi2-qwen36.sh /opt/supervisor-scripts/; chmod +x /opt/supervisor-scripts/qgi2-*.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
supervisorctl start qgi2-w27b-dflash2 >/dev/null; up 18037 60 || exit 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
supervisorctl start qgi2-qwen36 >/dev/null; up 18036 90 || { grep -iE "error|OOM" /var/log/portal/qgi2-qwen36.log | tail -4 | cut -c1-200; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
supervisorctl start qgi2-dgemma >/dev/null; up 18041 90 || { grep -iE "error|OOM" /var/log/portal/qgi2-dgemma.log | tail -4 | cut -c1-200; exit 1; }
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tail -4; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tail -1
grep -c "memory allocation failed" /var/log/portal/qgi2-dgemma.log
for m in "18037 QGI-2-Worker-DFlash2" "18036 QGI-Flash-3.6" "18041 QGI-DGemma-26B"; do set -- $m
  curl -s -m 180 http://127.0.0.1:$1/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK_SENTINEL\"}],\"max_tokens\":512}" | python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('$2', repr((m.get('content') or '')[:60]), d['usage']['completion_tokens'])"; done
