#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "QGI-2 Worker DFlash2"
source /venv/sglang-main/bin/activate
export ROLE=traceable CUDA_VISIBLE_DEVICES=3
cd /root/qgi2-deploy
pty ./launch-qgi2-worker-27b.sh 2>&1
