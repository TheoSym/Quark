set -u
cp /root/launch-qwen36-35b-dflash.sh /root/qgi2-deploy/ && chmod +x /root/qgi2-deploy/launch-qwen36-35b-dflash.sh
cp /root/qgi2-qwen36.sh /opt/supervisor-scripts/ && chmod +x /opt/supervisor-scripts/qgi2-qwen36.sh
cp /root/qgi2-qwen36.conf /etc/supervisor/conf.d/
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
supervisorctl start qgi2-qwen36
for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:18036/v1/models >/dev/null && { echo "UP after $((i*10))s"; break; }; sleep 10; done
supervisorctl status qgi2-qwen36
curl -sf -m 3 http://127.0.0.1:18036/v1/models || { echo "--- log tail"; tail -n 40 /var/log/portal/qgi2-qwen36.log 2>/dev/null | grep -iE "error|Traceback|Exception|not supported|OOM" | tail -8; }
echo; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
echo "== DG fetch"; tail -c 120 /models/dg.fetch.log | tr '\r' '\n' | tail -1; du -sh /models/diffusiongemma-26B-A4B-it-NVFP4 2>/dev/null
echo "== vllm"; tail -n 2 /root/vllm-install.log | cut -c1-160
