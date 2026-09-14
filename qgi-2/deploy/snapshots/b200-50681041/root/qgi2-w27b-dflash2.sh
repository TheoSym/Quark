#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "QGI-2 Worker DFlash2"
source /venv/sglang-main/bin/activate
export ROLE=traceable CUDA_VISIBLE_DEVICES=3 MEM_FRAC=0.30   # 2026-09-13: was the launch default 0.45 (92 GB); 0.30 ≈ 55 GB leaves room for the two other tenants
cd /root/qgi2-deploy
pty ./launch-qgi2-worker-27b.sh 2>&1
