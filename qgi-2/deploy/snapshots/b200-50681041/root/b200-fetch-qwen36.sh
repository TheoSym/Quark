set -u
PY=/venv/sglang-main/bin/python
$PY -c "import huggingface_hub" 2>/dev/null || PY=/venv/main/bin/python
$PY -c "import huggingface_hub" || { echo "no huggingface_hub in any venv"; exit 1; }
cat > /root/fetch_qwen36.py <<'EOF'
from huggingface_hub import snapshot_download
for repo, dest in [("Qwen/Qwen3.6-35B-A3B-FP8", "/models/Qwen3.6-35B-A3B-FP8"), ("z-lab/Qwen3.6-35B-A3B-DFlash", "/models/Qwen3.6-35B-A3B-DFlash")]:
    print("==", repo, flush=True); snapshot_download(repo, local_dir=dest, max_workers=16)
print("FETCH_DONE", flush=True)
EOF
nohup $PY /root/fetch_qwen36.py > /models/qwen36.fetch.log 2>&1 &
sleep 20; tail -c 400 /models/qwen36.fetch.log; echo; du -sh /models/Qwen3.6-35B-A3B-FP8 2>/dev/null
