set -u
O=/root/qgi2-src; rm -rf $O; mkdir -p $O/supervisor $O/scripts $O/etc
cp /etc/supervisor/conf.d/qgi2*.conf $O/supervisor/ 2>/dev/null; ls /etc/supervisor/conf.d/ > $O/supervisor/conf.d.ls
cp /opt/supervisor-scripts/qgi2-*.sh $O/scripts/
cp /workspace/CLAUDE.md /workspace/AGENTS.md $O/ 2>/dev/null
ls -la /root /root/*.sh /workspace /opt 2>/dev/null > $O/root.ls
for f in /root/*.sh /root/*.py /root/*.md /root/*.txt /root/*.env /root/*.conf; do [ -f "$f" ] && cp "$f" $O/scripts/ ; done
cp /etc/caddy/Caddyfile $O/etc/ 2>/dev/null; cp -r /etc/caddy $O/etc/ 2>/dev/null
env | grep -E '^(VAST|WORKSPACE|OPEN_BUTTON|PROVISIONING|PORTAL|DATA_DIR|HF_HOME|ACTIVE_VENV|JUPYTER|CUDA|NVIDIA|PYTHON)' | sed -E 's/(TOKEN|KEY|SECRET)=.*/\1=REDACTED/' > $O/env.txt
# python env used by the engines
PID=$(supervisorctl pid qgi2-qwen38); ls -l /proc/$PID/exe > $O/python-exe.txt 2>&1
PY=$(readlink -f /proc/$PID/exe); echo "$PY" >> $O/python-exe.txt
tr '\0' '\n' < /proc/$PID/environ | grep -vE 'TOKEN|KEY|SECRET' > $O/engine-environ.txt
$PY -m pip freeze > $O/requirements.freeze.txt 2>&1
$PY -c "import torch,sglang;print('torch',torch.__version__,'cuda',torch.version.cuda);print('sglang',sglang.__version__)" > $O/versions.txt 2>&1
nvcc --version 2>/dev/null | tail -1 >> $O/versions.txt; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 >> $O/versions.txt
$PY -m pip show sglang flashinfer-python torch 2>/dev/null | grep -E '^(Name|Version|Location)' >> $O/versions.txt
# model provenance
for d in /models/*/; do n=$(basename $d); echo "== $n $(du -sh $d 2>/dev/null | cut -f1)"; ls $d | head -20 | tr '\n' ' '; echo; grep -o '"_name_or_path"[^,]*' $d/config.json 2>/dev/null; done > $O/models.txt
cat /models/*.fetch.log 2>/dev/null | grep -iE 'huggingface|repo|snapshot|hf ' | sort -u | head -40 > $O/models-fetch.txt
ls /root/.cache/huggingface/hub 2>/dev/null > $O/hf-hub.ls
# gpu usage per process
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/gpu-apps.csv
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader > $O/gpu.csv
for p in $(awk -F', ' '{print $2}' $O/gpu-apps.csv | sort -u); do echo "$p $(tr '\0' ' ' < /proc/$p/cmdline | grep -oE 'served-model-name [^ ]+|sglang::[^ ]+' | head -1) $(grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' /proc/$p/environ 2>/dev/null | tr -d '\0' | head -1)"; done > $O/gpu-pids.txt
tar czf /root/qgi2-src.tgz -C /root qgi2-src && ls -l /root/qgi2-src.tgz
