#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "QGI-Flash 3.6"
set -euo pipefail   # after the base-image helpers, which are not strict-mode clean
source /venv/sglang-main/bin/activate
export CUDA_VISIBLE_DEVICES=3 PORT=18036 NAME=QGI-Flash-3.6 MEM_FRAC=0.55   # 0.45 fails: "Not enough GPU memory for hybrid state cache" (the pool must cover weights + KV + GDN slots)
cd /root/qgi2-deploy
pty ./launch-qwen36-35b-dflash.sh 2>&1
