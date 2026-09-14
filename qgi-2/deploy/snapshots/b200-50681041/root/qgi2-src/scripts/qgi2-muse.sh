#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Muse Glimmer"
source /venv/sglang-main/bin/activate
export ROLE=muse CUDA_VISIBLE_DEVICES=3 MEM_FRAC=0.72
cd /root/qgi2-deploy
pty ./launch-qgi2-worker-27b.sh 2>&1
