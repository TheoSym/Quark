# Self-hosting QGI-2's models

The spec's architecture only works when you own the processes: speculation is
fixed at launch, HiCache is a launch flag, and prefix caching is per-process.
Routing through a gateway gives you none of those. This directory holds the
launch scripts and units to bring the models up on the fleet's idle capacity.

Everything here follows `AI-Infra/.claude/skills/model-serve` — the `:18xxx` port
convention, `--host 127.0.0.1`, systemd persistence, and the **§7
agent-readiness gate**, which is mandatory and which two previous bringups
failed while passing every other smoke test.

## What QGI-2 needs, and what already exists

| Process | Role | Speculation | Status |
|---|---|---|---|
| `QGI-3.8-27b DFlash` | worker | DFlash2 n=7 | **running** — vidatron GPU1, SGLang :18031 |
| planner | planner | MTP n=2 | **to host** — vidatron GPU0, 96 GB idle |
| `QGI-Embed` | embedder | — | **running** — iridtron Ollama |

Three processes, every one of them speculating, and that is the whole
deployment. An earlier revision of this file called for a fourth — an MTP
worker for the `deterministic` profile, on the spec's claim that DFlash2
cannot produce greedy output. That claim is measured false on this exact model
(syv-ai/qwen38-27b-rtx3090: DFlash2 greedy at 2.90–3.45 tokens/step), so the
running worker serves all three profiles:

```
$ qgi2 --config ../config/qgi2.qgi-fleet.toml doctor --profile deterministic
  route      worker   dflash2 n=7    -> vidatron:18031 (sglang)
  tool_args  worker   dflash2 n=7    -> vidatron:18031 (sglang)
  extract    worker   dflash2 n=7    -> vidatron:18031 (sglang)
```

`launch-qgi2-worker-mtp.sh` and its unit stay in this directory as an
**optional second worker** for throughput (c8+ is where MTP's 1.97× beats
DSpark/DFlash2's serial-regime lead), not as a profile requirement.

### Swapping the worker's speculator

The running worker is DFlash2. If you relaunch it, the measured ranking on
this hardware (SamSammane/qwen38-27b-nvfp4-sm121-vllm, 17/17 runs) is
**DSpark K=7** for QGI-2's regime — 3.5 acceptance thinking-on, ~2.1 off, and
200–223 tok/s single-stream on the RTX PRO 6000. QGI-2 declares it as
`speculation = "dspark"`; the launch line comes from
SamSammane/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark. Either way, size the GDN
state pool first — see below.

### The GDN state pool (hybrid models only)

The 27B is a hybrid: each running request pins recurrent-state slots for its
lifetime, no paging, and a slot is 78.4 MB. Speculation multiplies it — a
draft block of 7 wants 8 draft slots on top of the 4 the lazy radix strategy
keeps, so 8 concurrent requests × 12 slots ≈ 7.5 GB before a single KV page.
The pool caps concurrency independently of KV, and SGLang will not tell you
until it refuses to start.

Declare it and `qgi2 hicache` prints the pool, what it leaves for KV, and the
flags that pin it (`--max-mamba-cache-size`, `--mamba-radix-cache-strategy
extra_buffer_lazy`):

```toml
[hicache.gdn]
slot_bytes = 78400000
lazy_slots = 4
max_concurrent = 8
speculation = { method = "dspark", n = 7 }
budget = { total_gb = 96.0, mem_fraction_static = 0.90, weights_gb = 24.5, runtime_gb = 3.5 }
```

## Where things fit

| Host | GPU | Free | Proposed |
|---|---|---|---|
| vidatron-G192 | 2× RTX PRO 6000 Blackwell 96 GB | **GPU0 idle 96 GB** | planner |
| vegatron-G144 | 3× RTX A6000 48 GB | **GPU0 idle 48 GB** | optional second worker |
| iridtron-G96 | 1× RTX PRO 6000 96 GB | ~69 GB | gateway + embedder; leave it alone |

Putting the planner on vidatron GPU0 keeps it on the same box as the worker, so
the planner→worker hop inside a turn stays on the PCIe bus rather than the
tailnet, and both share one host-memory pool for HiCache L2.

### One hardware constraint that will bite

**vegatron is Ampere (A6000, sm_86) and cannot run FP8.** The parked
`Qwen3.8-27B` weights are FP8, so `systemctl enable --now qwen38-27b` will bring
up a process that fails on the first forward pass. The recipe that fits is
syv-ai/qwen38-27b-rtx3090's **W4A16 AutoRound** — it serves the 27B with DFlash2
on a 24 GB 3090, so a 48 GB A6000 is comfortable. Two of its flags are not
optional on a hybrid model: `--mamba-cache-mode align` (without it the prefix
cache never hits) and `--enable-force-include-usage` (without it streamed
replies carry no `cached_tokens`, and QGI-2's cache metric reads zero). Both
are in `qgi2`'s vLLM launch hint.

Only do this if you want a second worker at all — see the process table.

## Order of operations

```bash
# 1. Clear the GPU (model-serve §0 — a crashed server pins VRAM via worker procs)
./clear-gpu.sh

# 2. Launch, wait for liveness, then run the §7 gate. Do not skip the gate:
#    HTTP 200 is not readiness, and a model without the reasoning/tool parsers
#    dumps raw <function=...> XML into content and returns zero function_call
#    items — which is exactly what QGI-2's tool-args step consumes.
sudo cp qgi2-planner.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now qgi2-planner
./verify-agent-ready.sh 18033 "QGI-2-Planner"

# 3. Point QGI-2 at it and confirm every step routes
qgi2 --config ../config/qgi2.qgi-fleet.toml doctor
```

## Reaching the backends from QGI-2

Backends bind `127.0.0.1`, so QGI-2 cannot dial `100.x.x.x:18033` directly
unless the port is exposed. Two options:

- **`tailscale serve`** the port on each host, and check
  `infra/tailscale-acl.hujson` first — gpu-lane→gpu-lane is port-listed, and an
  ACL miss looks exactly like a bind problem. This is the low-latency path.
- **Through LiteLLM** (`:4001`), which is already exposed. Simpler, but adds a
  hop and puts the Cloudflare edge in the path. QGI-2 streams by default
  precisely so that a long planner step does not hit the edge's 524.

## Secrets

`model-serve` §7: never put `HF_TOKEN` in a `.service` file — it is
world-readable and prints in `systemctl cat`. The units here use
`EnvironmentFile=/etc/qgi2-<name>.env`, mode 0600, root-owned.

```bash
sudo install -m600 /dev/null /etc/qgi2-planner.env
sudo tee /etc/qgi2-planner.env >/dev/null <<'EOF'
HF_TOKEN=hf_...
EOF
```
