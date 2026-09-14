set -u
python3 - <<'EOF'
import yaml
p="/etc/portal.yaml"; d=yaml.safe_load(open(p)); a=d["applications"]
a.pop("Muse Glimmer", None)
a["QGI-Flash 3.6"]={"hostname":"localhost","external_port":10400,"internal_port":18036,"open_path":"/v1/models","name":"QGI-Flash 3.6"}
a["QGI DGemma 26B"]={"hostname":"localhost","external_port":10600,"internal_port":18041,"open_path":"/v1/models","name":"QGI DGemma 26B"}
open(p+".tmp","w").write(yaml.safe_dump(d,sort_keys=False)); import os; os.replace(p+".tmp",p); print("portal.yaml:", list(a))
EOF
cp /root/launch-diffusiongemma-26b.sh /root/qgi2-deploy/ && chmod +x /root/qgi2-deploy/launch-diffusiongemma-26b.sh
cp /root/qgi2-dgemma.sh /opt/supervisor-scripts/ && chmod +x /opt/supervisor-scripts/qgi2-dgemma.sh
cp /root/qgi2-dgemma.conf /etc/supervisor/conf.d/
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
supervisorctl start qgi2-qwen36
for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:18036/v1/models >/dev/null && { echo "qwen36 UP after $((i*10))s"; break; }; sleep 10; done
supervisorctl status qgi2-qwen36
if ! curl -sf -m 3 http://127.0.0.1:18036/v1/models >/dev/null; then echo "--- qwen36 log"; tail -n 60 /var/log/portal/qgi2-qwen36.log 2>/dev/null | grep -iE "error|Traceback|Exception|not supported|OOM|ValueError" | tail -8 | cut -c1-220; exit 1; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
supervisorctl start qgi2-dgemma
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:18041/v1/models >/dev/null && { echo "dgemma UP after $((i*10))s"; break; }; sleep 10; done
supervisorctl status qgi2-dgemma
curl -sf -m 3 http://127.0.0.1:18041/v1/models >/dev/null || { echo "--- dgemma log"; tail -n 80 /var/log/portal/qgi2-dgemma.log 2>/dev/null | grep -iE "error|Traceback|Exception|not supported|OOM|ValueError" | tail -10 | cut -c1-220; }
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep 68ee8f22
for p in 18036 18041; do tailscale serve --bg --tcp $p tcp://127.0.0.1:$p >/dev/null; done; tailscale serve status | grep -c "127.0.0.1"
