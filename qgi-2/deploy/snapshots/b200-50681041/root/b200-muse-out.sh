set -u
supervisorctl stop qgi2-muse
rm -f /etc/supervisor/conf.d/qgi2-muse.conf /opt/supervisor-scripts/qgi2-muse.sh
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
rm -rf /models/Muse-Glimmer-NVFP4 /models/Muse-Glimmer-30B-assistant /models/muse.fetch.log
sleep 3; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h / | tail -1
mkdir -p /models
nohup sh -c 'hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir /models/Qwen3.6-35B-A3B-FP8 && hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir /models/Qwen3.6-35B-A3B-DFlash && echo FETCH_DONE' > /models/qwen36.fetch.log 2>&1 &
echo download started
