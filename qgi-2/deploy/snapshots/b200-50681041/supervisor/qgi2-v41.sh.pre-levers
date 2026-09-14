#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "QGI-2 V41"
source /venv/sglang/bin/activate
export MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4.1-Flash}"
export PORT=18034 NAME=QGI-2-V41 CUDA_VISIBLE_DEVICES="0,1" TP="2" HICACHE=1 MAX_RUNNING=32 MEM_FRAC=0.92 MAX_TOTAL_TOKENS=4000000
cd /root/qgi2-deploy
pty ./launch-qgi2-planner-v41.sh 2>&1
