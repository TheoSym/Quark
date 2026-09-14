#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Qwen3.8 Flash Next"
source /venv/sglang-main/bin/activate
export MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-NVFP4-RadixArk}" EXPORT=radixark
export PORT=18035 NAME=Qwen3.8-Flash-Next-NVFP4 CUDA_VISIBLE_DEVICES="2" MEM_FRAC=0.85 KV_DTYPE=fp8_e4m3 MTP=1
cd /root/qgi2-deploy
pty ./launch-qwen38-flash-next.sh 2>&1
