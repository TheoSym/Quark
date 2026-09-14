---
pipeline_tag: text-generation
library_name: transformers
base_model:
  - Qwen/Qwen3.6-35B-A3B
license: apache-2.0
inference: false
tags:
  - dflash
  - speculative-decoding
  - speculative-decoding-draft
  - block-diffusion
  - draft-model
  - diffusion-language-model
  - efficiency
  - qwen
  - qwen3
  - qwen3.6
  - sglang
---

# Qwen3.6-35B-A3B-DFlash

[Paper](https://arxiv.org/abs/2602.06036) | [Github](https://github.com/z-lab/dflash) | [Blog](https://z-lab.ai/projects/dflash)

This DFlash draft model is a joint retrain from [Z-Lab](https://z-lab.ai) and [Modal](https://modal.com), trained with 40k sequence length and sliding-window attention for improved long-context performance. It is mirrored across the following Hugging Face repositories:

- [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
- [`modal-labs/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/modal-labs/Qwen3.6-35B-A3B-DFlash)

This repository contains a DFlash draft model for `Qwen/Qwen3.6-35B-A3B`. It is not a standalone language model. It is intended to be paired with the target model in a speculative decoding server.

DFlash uses a lightweight block diffusion draft model to propose multiple tokens in parallel. The target model verifies those proposals, improving serving throughput while preserving the target model's output distribution.

<div align="center">
  <img src="assets/dflash_system.png" alt="DFlash Architecture" width="85%">
</div>

## Quick Start

### Installation

#### SGLang

Install a recent SGLang build with DFlash support:

```bash
uv pip install --upgrade "sglang[all]"
```

For best performance on Blackwell GPUs, use an SGLang build that includes DFlash, FA4/TRT-LLM attention, and FlashInfer support.

#### vLLM

For vLLM support, please refer to [vllm-project/vllm#40898](https://github.com/vllm-project/vllm/pull/40898). We will update the PR to make it merge-ready soon.

### Launch Server

This model should be used with an inference server that supports DFlash speculative decoding. An example SGLang deployment is:

```bash
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend fa4 \
  --attention-backend trtllm_mha \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-scheduler-strategy extra_buffer \
  --tp-size 1 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --cuda-graph-backend-prefill tc_piecewise \
  --enable-flashinfer-allreduce-fusion \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

Block size `8` is the recommended default for higher-concurrency serving. Block size `16` gives longer accept lengths and strong concurrency-1 throughput in most workloads.

## Benchmark Results

We benchmarked DFlash against the autoregressive baseline and Qwen's built-in MTP draft path. DFlash reaches up to `3.61x` speedup at concurrency 1 and `2.89x` at concurrency 32. Across the benchmark suite, DFlash delivers higher throughput than MTP at every matched setting where both completed.

### Setup

- Runtime: SGLang on 1x NVIDIA B200 GPU, tensor parallel size 1, `bfloat16`
- Backends: `trtllm_mha` target attention, `fa4` DFlash draft attention, `flashinfer` linear-attention prefill and decode
- Workloads: GSM8K, MATH500, HumanEval, MBPP, and MT-Bench with the Qwen chat template
- Decoding: greedy, thinking enabled, max output length 4096 tokens
- Measurement: 5 independent runs per configuration at concurrency 1 and 32 with continuous batching
- Throughput: generated output tokens / wall-clock benchmark time, including prefill and scheduling
- Accept length: `completion_tokens / spec_verify_ct` per generation turn, averaged across generation turns

### Throughput and Speedup

Each cell is `output tok/s (speedup)`. Bold marks the fastest speculative configuration in each row.

#### Concurrency 1

| Workload | Baseline | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 308.6 (1.00x) | 610.5 (1.98x) | 688.7 (2.23x) | 632.6 (2.05x) | 895.0 (2.90x) | 485.9 (1.57x) | **914.6 (2.96x)** |
| math500 | 309.3 (1.00x) | 641.8 (2.08x) | 726.2 (2.35x) | 699.8 (2.26x) | 1011.6 (3.27x) | 561.8 (1.82x) | **1116.3 (3.61x)** |
| humaneval | 306.4 (1.00x) | 607.7 (1.98x) | 709.0 (2.31x) | 631.6 (2.06x) | 943.2 (3.08x) | 488.7 (1.60x) | **1008.9 (3.29x)** |
| mbpp | 307.1 (1.00x) | 597.3 (1.94x) | 696.5 (2.27x) | 594.4 (1.94x) | 889.3 (2.90x) | 443.5 (1.44x) | **912.8 (2.97x)** |
| mt-bench | 306.2 (1.00x) | 555.3 (1.81x) | 614.4 (2.01x) | 526.9 (1.72x) | **711.5 (2.32x)** | 381.4 (1.25x) | 686.5 (2.24x) |

#### Concurrency 32

| Workload | Baseline | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 3495.1 (1.00x) | 6271.3 (1.79x) | 7196.1 (2.06x) | 6769.7 (1.94x) | **8786.4 (2.51x)** | 5439.1 (1.56x) | 8168.1 (2.34x) |
| math500 | 3494.8 (1.00x) | 6745.6 (1.93x) | 7582.3 (2.17x) | 7751.9 (2.22x) | 9991.2 (2.86x) | 6516.8 (1.86x) | **10106.8 (2.89x)** |
| humaneval | 3507.0 (1.00x) | 6417.9 (1.83x) | 7494.7 (2.14x) | 6976.1 (1.99x) | **9511.1 (2.71x)** | 5575.9 (1.59x) | 9055.4 (2.58x) |
| mbpp | 3570.1 (1.00x) | 6248.7 (1.75x) | 7403.6 (2.07x) | 6546.8 (1.83x) | **9074.6 (2.54x)** | 5112.1 (1.43x) | 8274.1 (2.32x) |
| mt-bench | 3244.8 (1.00x) | 5428.4 (1.67x) | 5933.1 (1.83x) | 5435.0 (1.67x) | **6591.7 (2.03x)** | 4213.6 (1.30x) | 5692.0 (1.75x) |

### Accept Length

Mean accept length at concurrency 1. Bold marks the higher value in each matched MTP/DFlash pair.

| Workload | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | **3.474** | 3.455 | 5.288 | **5.404** | 6.453 | **6.954** |
| math500 | **3.559** | 3.553 | 5.522 | **5.743** | 6.840 | **7.629** |
| humaneval | 3.369 | **3.459** | 4.952 | **5.352** | 5.878 | **6.797** |
| mbpp | 3.280 | **3.358** | 4.613 | **4.973** | 5.292 | **6.052** |
| mt-bench | **3.135** | 3.075 | **4.365** | 4.326 | 5.061 | **5.118** |

## Citation

If you find DFlash useful, please cite the original paper:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
