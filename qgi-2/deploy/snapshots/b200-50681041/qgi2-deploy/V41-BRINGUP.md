# Bringing up QGI-2 on DeepSeek-V4.1-Flash, single-model, 8× RTX PRO 6000

The plan of record for the first live run. One process serves every step;
the 27B worker stays on the shelf until the single-model numbers say it earns
a card back.

## Layout

| GPUs | Process | Model | Speculation | Steps |
|---|---|---|---|---|
| 0–3 | `qgi2-v41` | DeepSeek-V4.1-Flash, TP4 EP4, Engram tables in host RAM, text only | DSpark block 5 | route, plan, tool-args, extract, answer |
| 4–7 | spare | second replica later (effort racing, extract‖answer pipelining), or the 27B worker if measured cheaper per step | | |

No embedder. Retrieval runs lexical-only and every response carries
`retrieval_degraded`; add a 2 GB embedder on card 4 only if graph misses show
up on queries that do not name a node.

## Host requirements, checked before downloading anything

| Need | Why | Check |
|---|---|---|
| ~600 GB free on NVMe | the checkpoint is 510 GB plus download temp files | `df -h /data` |
| ≥ 300 GB host RAM | Engram tables (~189 GiB) move to host memory to fit four cards, HiCache L2 wants the rest | `free -g` |
| 4 free 96 GB cards | ~320 GB resident weights at TP4 after Engram is off-GPU, plus KV | `nvidia-smi` |
| SGLang preview build | V4.1-Flash is not in a released SGLang; use `lmsysorg/sglang:dev-dsv41` or a build from its branch | image pulls |

**Unverified:** RTX PRO 6000 (SM120) is not on the cookbook's list
(GB300/H200/B200/B300). The MXFP4 MoE runner is the part to watch on first
load. If it refuses, the fallback is DeepSeek-V4-Flash at 168 GB on two cards
with the same harness config.

## The vast.ai instance (50403223)

The box is a vast.ai rental: 8× RTX PRO 6000 WS, 755 GB RAM, 320 cores,
1.6 TB on the container overlay, a 10 GB volume at `/workspace` (too small for
weights — they live at `/models` on the overlay, which does **not** survive a
recycle or destroy; stop/start is fine). It is an unprivileged container: no
Docker-in-Docker, so the SGLang preview image cannot be run — its exact build
is reproduced from source instead (below).

- **SSH:** `ssh -i ~/.ssh/ai-infra -p 32803 root@70.69.192.6` (direct port;
  the ssh1.vast.ai proxy drops the handshake). The account's default
  `id_ed25519` key is passphrase-protected and fails silently in batch mode —
  use the `ai-infra` key, which is also in the instance's authorized_keys.
- **SGLang:** `/venv/sglang`, source at `/root/sglang` checked out at commit
  `824bb45e` — the `ai.sglang.build.commit` label of
  `lmsysorg/sglang:dev-cu12-dsv41` — with the image's pins
  (`sgl-kernel==0.4.6.post1`, `flashinfer-python==0.6.18`). Install log:
  `/root/sglang-install.log`.
- **Weights:** `/models/<repo basename>` via `/root/qgi2-deploy/fetch-weights.sh`;
  log at `/models/fetch.log`; `.qgi2-pin.json` beside each checkpoint.
- **Service:** supervisor program `qgi2-v41` (wrapper
  `/opt/supervisor-scripts/qgi2-v41.sh`, `autostart=false` until the first
  manual launch succeeds). `supervisorctl start qgi2-v41`; logs at
  `/var/log/portal/qgi2-v41.log`.
- **Reaching it:** SGLang binds `127.0.0.1:18034`. It is published through the
  Caddy token edge as external port 10100 → `http://70.69.192.6:32819/v1`,
  auth `Authorization: Bearer $OPEN_BUTTON_TOKEN` (the instance's portal token;
  put it in the qgi2 config's `api_key`). For a private path with no token,
  SSH-forward instead: `ssh -i ~/.ssh/ai-infra -p 32803 -L 18034:127.0.0.1:18034 root@70.69.192.6`
  and point the config at `http://127.0.0.1:18034/v1`.

### What the first day on that instance found (2026-09-11)

- **Host driver bug, blocking:** any single GPU or any *pair* initialises CUDA,
  any *three or more* fail with `cuInit` error 3. The driver's unified-memory
  layer initialises and immediately tears down. Host runs the open kernel
  module 590.48.01 on kernel 6.8; this matches
  [NVIDIA/open-gpu-kernel-modules#858](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/858),
  whose only known fix is `rmmod nvidia_uvm && modprobe nvidia_uvm` **on the
  host** — not possible from the container, and a container reboot does not
  clear it. A 4-card TP planner cannot start on this host until the host
  owner reloads the module or vast moves the instance.
- **SM120 works otherwise.** With `FLASHINFER_CUDA_ARCH_LIST=12.0a`, the
  reproduced dsv41 build served Qwen3.8-27B-FP8 on one card: FlashInfer
  attention and sampler JIT, FP8 GEMMs, the hybrid linear-attention (GDN)
  path, and CUDA graph capture all ran, and it answered a coding prompt
  correctly. Without the variable, FlashInfer probes every GPU on the host,
  trips the bug above, and rejects the card as "below sm75".
- **Qwen3.5-family checkpoints need `--language-only`** (not
  `--language-model-only`, which only accepts a different architecture);
  otherwise the multimodal processor load fails.
- **Install gotchas:** `uv` with `UV_NO_CACHE=1` (the image default) stalled
  silently on the big torch wheel — set a cache dir; the source build needs
  `SGLANG_BUILD_RUST_EXTS=none` unless a Rust toolchain is present; the kernel
  package is now called `sglang-kernel` and is pulled in by the main install.
- **The git fetch of a bare commit** from GitHub takes 10+ minutes; it does
  arrive.

## Steps

```bash
cd qgi-2/deploy

# 1. Pinned weights. models.lock.json names the revision and every sha256;
#    fetch-weights.sh refuses to start without the disk space and verifies
#    every digest after the download. Regenerate the lock only with
#    pin-weights.py, after a deliberate model change.
MODELS_DIR=/data/models ./fetch-weights.sh planner        # ~510 GB
MODELS_DIR=/data/models ./fetch-weights.sh worker         # ~31 GB, optional now
MODELS_DIR=/data/models ./fetch-weights.sh worker_drafter # ~4 GB, optional now

# 2. Launch. TP/EP, DSpark, Engram-in-host, parsers and HiCache are all in
#    the script; MODEL_PATH is the only required input.
./clear-gpu.sh
MODEL_PATH=/data/models/DeepSeek-V4.1-Flash ./launch-qgi2-planner-v41.sh
#    (or install qgi2-v41 as a systemd unit with EnvironmentFile, as the
#    other units here do)

# 3. The §7 gate. HTTP 200 is not readiness; this checks tool calls come back
#    as function_call items, constrained JSON decodes, and /metrics exposes
#    the cache and acceptance gauges the harness reads.
./verify-agent-ready.sh 18034 QGI-2-V41

# 4. Point the harness at it, on the machine that runs jcode.
cp ../config/qgi2.v41-single.toml ~/.qgi2/config.toml   # then fill in the host
qgi2 doctor        # every step must route to :18034 with dspark n=5
qgi2 plan          # prints "single-model" and the per-step table
qgi2 serve
qgi2 config --jcode >> ~/.jcode/config.toml
jcode --provider qgi2

# 5. The numbers. Two turns is enough to see the cache metric; the bench is
#    what turns projections into measurements on coding tasks.
curl -s localhost:8788/qgi2/metrics | jq
python3 ../bench/ab_compare.py --baseline-only    # stock jcode, same model
python3 ../bench/ab_compare.py                    # through QGI-2
```

## What the harness does differently on this model

- **Speculation is declared, not tabled.** The router's table says MTP for the
  planner; the config declares `dspark`/5 and the override path already
  existed for cloud planners. `qgi2 doctor` and `serve` build the same router
  the session runs.
- **Reasoning effort per step.** V4.1 has a 1–100 dial instead of a thinking
  switch. The router now states an effort for every model step from the
  profile: structured steps 5, plan 40, answer 70 (Quick: 1/10/20). It is
  sent only to endpoints with `reasoning_effort = true`, so Qwen and gateway
  routes never see the field.
- **Single-model mode** is detected from both roles resolving to one URL and
  model. The planner:worker ratio metric is suppressed; cache hit rate,
  acceptance and rejection rate still apply.

## Order of the follow-on work (see the plan table in the session notes)

1. codebase-memory-mcp registered as a jcode tool — no harness change.
2. Line-store memory (`qgi2-memory`): add at window exit, deterministic
   retrieve into segment 5, last two turns verbatim in segment 6.
3. Compaction event, rank-cut with a counter first.
4. Extract ‖ answer pipelining on a second replica (cards 4–7).
5. Effort racing: effort 20 vs 80 on two replicas, rules verify the cheap one.
