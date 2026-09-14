# Reproducing the B200 box (vast.ai instance 50681041)

Captured 2026-09-14, minutes before the instance was destroyed. Everything in
this directory came off the box; nothing here is reconstructed from memory.
Weights were not copied (about 850 GB); every checkpoint is pinned below and
re-downloads from the Hub.

## What was running at capture

Five engine processes, all SGLang except the last, all also exposed on the
tailnet via `tailscale serve` (node `qgi-g732`). Exact command lines, as the
kernel saw them, are in `env/live-processes.txt`. That file is the truth when
a launch script and a supervisor wrapper disagree.

| Port | Served name | Checkpoint | Card(s) | Engine venv | Speculation |
|---|---|---|---|---|---|
| 18034 | `QGI-2-V41` | DeepSeek-V4.1-Flash | 0,1 (TP2, EP2) | `/venv/sglang` @ `824bb45e` | DSpark block 5, HiCache |
| 18035 | `Qwen3.8-Flash-Next-NVFP4` | Qwen3.8-Flash-Next-NVFP4-**RadixArk** | 2 | `/venv/sglang-main` @ `7c195b91` | NEXTN 3/1/4, fp8 KV |
| 18037 | `QGI-2-Worker-DFlash2` | Qwen3.8-27B-FP8 + DFlash2 drafter | 3 (mem 0.30) | `/venv/sglang-main` | DFLASH 8 drafts, HiCache |
| 18036 | `QGI-Flash-3.6` | Qwen3.6-35B-A3B-FP8 + DFlash drafter | 3 (mem 0.55) | `/venv/sglang-main` | DFLASH 8 drafts |
| 18041 | `QGI-DGemma-26B` | diffusiongemma-26B-A4B-it-NVFP4 | 3 (util 0.17) | `/venv/vllm` | none |

Note the 18035 server had been switched from the nvidia export to the
RadixArk export (`EXPORT=radixark` in `supervisor/qgi2-qwen38.sh`); the
served name did not change. `root/spec_ab.sh` is the head-to-head that
decided it.

## Host

- vast.ai image `vastai/base-image:cuda-13.2.1-auto` (unprivileged container,
  no Docker, supervisor + Caddy portal). 4× B200 183 GB (SM100), driver
  595.91.07, CUDA 13.2.1, kernel 7.0 aws, 2 TB RAM, 192 cores. Full detail:
  `env/nvidia-smi-q.txt`, `env/env.txt`, `env/os-release.txt`.
- Do **not** set `FLASHINFER_CUDA_ARCH_LIST` here; B200 is `10.0a` and
  auto-detect is right. That variable was the SM120 workaround.

## Checkpoints

Pinned with sha256 in `../../models.lock.json` (planner, worker, drafter) and
fetched by `../../fetch-weights.sh <role>`. The rest were `snapshot_download`
at whatever the Hub served on 2026-09-12/13; `models/*.files.tsv` lists every
file with its byte size so a re-download can be checked, and `models/models/*`
holds each checkpoint's config, quant config, chat template and README.

| Directory on box | Hub repo | Revision |
|---|---|---|
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Qwen3.8-27B-FP8 | `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Qwen3.8-27B-DFlash2 | `incoai/Qwen3.8-27B-DFlash2` | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Qwen3.8-27B-DSpark | `RadixArk/Qwen3.8-27B-DSpark` | unpinned |
| Qwen3.8-27B-NVFP4-RadixArk | `RadixArk/Qwen3.8-27B-NVFP4` | unpinned (`conversion-manifest.json` kept) |
| Qwen3.8-Flash-Next-NVFP4 | `nvidia/Qwen3.8-Flash-Next-NVFP4` | unpinned |
| Qwen3.8-Flash-Next-NVFP4-RadixArk | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | unpinned (`conversion_environment.json`: modelopt 0.46.0) |
| Qwen3.6-35B-A3B-FP8 | `Qwen/Qwen3.6-35B-A3B-FP8` | unpinned |
| Qwen3.6-35B-A3B-DFlash | `z-lab/Qwen3.6-35B-A3B-DFlash` | unpinned |
| diffusiongemma-26B-A4B-it-NVFP4 | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | unpinned |

Pin the unpinned ones with `../../pin-weights.py` on the next box before
anything is measured against them.

## Engines

Three venvs, built with `uv` (install logs: `root/sglang-install.log`,
`root/sglang-main-install.log`, `root/vllm-install.log`; exact package sets:
`venvs/*.freeze`).

1. **`/venv/sglang`** — SGLang at `824bb45e` (`sglang/sglang.HEAD`), the
   only tree with the DeepSeek-V4.1 classes. Built from source with
   `SGLANG_BUILD_RUST_EXTS=none`. Carries one uncommitted change,
   `sglang/sglang.working-tree.diff` (26 lines): the token-first prompt fix
   for `/v1/responses` on multimodal-encoder models, without which every
   Responses-API call to V4.1 fails with "texts cannot be empty". Script form:
   `../../patches/sglang-dsv41-responses-token-first-prompt.py`.
2. **`/venv/sglang-main`** — SGLang main at `7c195b91` (2026-09-12), needed
   for `qwen4_exp` (Flash-Next) and the DFLASH speculator. Same torch /
   flashinfer / sgl-kernel pins as (1). Has no V4.1 classes, so the planner
   cannot move here.
3. **`/venv/vllm`** — for DiffusionGemma only.

Python 3.12; torch 2.13 / flashinfer 0.6.18 / sgl-kernel 0.4.6.post1 per the
freezes.

## Bringing it up

```bash
# 1. weights (planner ~510 GB; the box pulled at ~5 GB/s)
MODELS_DIR=/models ../../fetch-weights.sh planner
MODELS_DIR=/models ../../fetch-weights.sh worker
MODELS_DIR=/models ../../fetch-weights.sh worker_drafter
# the rest: root/fetch_qwen36.py, root/fetch_dg.py, and hf download for the RadixArk/nvidia repos

# 2. venvs — replay the install logs' resolved sets from venvs/*.freeze
# 3. apply sglang/sglang.working-tree.diff to /root/sglang

# 4. optional but worth it: restore the JIT/autotune caches BEFORE first launch.
#    First boot was ~35 min (V41) / ~15 min (Flash-Next) of FlashInfer JIT +
#    TileLang + autotune; with the caches, restarts are ~5 min.
tar -C /root -xzf ../b200-jit-caches.tgz    # .cache/{flashinfer,sglang,vllm} .tilelang .triton
#    Only valid for the same torch/flashinfer/sglang pins on SM100.

# 5. services
cp supervisor/qgi2-*.conf /etc/supervisor/conf.d/
cp supervisor/qgi2-*.sh   /opt/supervisor-scripts/   # they cd /root/qgi2-deploy = ../../
cp portal/portal.yaml /etc/portal.yaml               # Caddy entries: external 10100/10200/10300/10400/10600
supervisorctl reread && supervisorctl update
supervisorctl start qgi2-v41        # then ../../verify-agent-ready.sh 18034 QGI-2-V41
supervisorctl start qgi2-qwen38 qgi2-w27b-dflash2 qgi2-qwen36
```

Card 3 is shared three ways; `supervisor/qgi2-w27b-dflash2.sh` and
`qgi2-qwen36.sh` carry the memory fractions that were found to coexist and
the failure that a lower one produced.

## Results that were measured on this box

- **Planner speed levers** (`root/ds_levers.sh`, `root/ds_speed.py`,
  `root/ds_levers.log`): baseline TTFT 0.18 s on a 7.7 K prompt, 254 tok/s
  single-stream, 1337 tok/s at 8 streams, DSpark accept 2.85. Q8KV8 sparse
  prefill and DeepEP both slightly slower; neither adopted.
- **Flash-Next export A/B** (`root/spec_ab.sh`): nvidia vs RadixArk NVFP4;
  RadixArk won and is what 18035 served.
- **Flash-Next KV/MTP trials** (`root/qwen38-*.log`): fp8 vs bf16 KV, MTP
  on/off, RadixArk. MTP accept 1.2–1.7 of 4 with either KV dtype.
- **27B worker under the two profiles** (`root/w27b-traceable.log`,
  `root/w27b-deterministic.log`).
- **Per-service engine logs** (`logs/qgi2-*.log`, last 4000 lines each):
  pool sizes, autotune, cache reports.

## Not captured

- Weights (see above), `/root/.ssh`, `/root/.tailscale`, `.vast_api_key`.
  The two tokens that appeared in copied inventory files were redacted.
- The vast `OPEN_BUTTON_TOKEN` and public ports (37972, 37259, …) are
  per-instance and die with it.
