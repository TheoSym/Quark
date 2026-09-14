set -u
echo "== fetch"; tail -c 200 /models/qwen36.fetch.log | tr '\r' '\n' | tail -2; du -sh /models/Qwen3.6-35B-A3B-FP8 /models/Qwen3.6-35B-A3B-DFlash 2>/dev/null
echo "== sglang-main arch support"
grep -rl "Qwen3_5MoeForConditionalGeneration" /root/sglang-main/python/sglang/srt/models/ | head -3
grep -rn "DFlashDraftModel\|class DFlash" /root/sglang-main/python/sglang/srt/models/*.py | head -3 | cut -c1-140
grep -rn "language.only\|language_only" /root/sglang-main/python/sglang/srt/server_args.py | head -2 | cut -c1-140
echo "== vllm venv"
if [ -x /venv/vllm/bin/python ] && /venv/vllm/bin/python -c "import vllm" 2>/dev/null; then /venv/vllm/bin/python -c "import vllm;print('vllm',vllm.__version__)"; else
  export UV_CACHE_DIR=/root/.uv-cache
  nohup sh -c 'uv venv --python 3.12 /venv/vllm && uv pip install --python /venv/vllm/bin/python vllm --torch-backend=cu130 && /venv/vllm/bin/python -c "import vllm;print(\"VLLM_OK\",vllm.__version__)"' > /root/vllm-install.log 2>&1 &
  echo "vllm install started"
fi
echo "== DG download"
PY=/venv/sglang-main/bin/python
cat > /root/fetch_dg.py <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/diffusiongemma-26B-A4B-it-NVFP4", local_dir="/models/diffusiongemma-26B-A4B-it-NVFP4", max_workers=16); print("FETCH_DONE", flush=True)
EOF
nohup $PY /root/fetch_dg.py > /models/dg.fetch.log 2>&1 &
echo started; free -g | head -2; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1
