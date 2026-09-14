#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "QGI DGemma 26B"
set -euo pipefail   # after the base-image helpers, which are not strict-mode clean
source /venv/vllm/bin/activate
export CUDA_VISIBLE_DEVICES=3 PORT=18041 NAME=QGI-DGemma-26B GPU_UTIL=0.17
cd /root/qgi2-deploy
pty ./launch-diffusiongemma-26b.sh --enable-auto-tool-choice --tool-call-parser gemma4 2>&1
