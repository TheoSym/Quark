set -u
echo "== qwen36"; curl -s -m 5 http://127.0.0.1:18036/v1/models | head -c 200; echo
cp /root/launch-diffusiongemma-26b.sh /root/qgi2-deploy/ && chmod +x /root/qgi2-deploy/launch-diffusiongemma-26b.sh
cp /root/qgi2-dgemma.sh /opt/supervisor-scripts/ && chmod +x /opt/supervisor-scripts/qgi2-dgemma.sh
supervisorctl start qgi2-dgemma 2>&1
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:18041/v1/models >/dev/null && { echo "dgemma UP after $((i*10))s"; break; }; sleep 10; done
supervisorctl status qgi2-dgemma
curl -sf -m 3 http://127.0.0.1:18041/v1/models >/dev/null || { echo "--- dgemma log"; grep -iE "error|Traceback|Exception|not supported|OOM|ValueError|No available memory" /var/log/portal/qgi2-dgemma.log | tail -12 | cut -c1-240; }
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep 68ee8f22
for p in 18036 18041; do tailscale serve --bg --tcp $p tcp://127.0.0.1:$p >/dev/null; done; tailscale serve status 2>/dev/null | grep -c "127.0.0.1"
