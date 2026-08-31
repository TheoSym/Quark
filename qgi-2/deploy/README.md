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
| worker-mtp | worker | MTP n=3 | **to host** — only for the `deterministic` profile |

The third process is what the spec means by *"two processes"* being a floor
rather than a total. `traceable` needs a DFlash2 worker and `deterministic`
needs an MTP one, because DFlash2 cannot produce greedy output. One deployment
cannot serve both profiles.

If you only ever run `traceable`, skip it — `qgi2 doctor` will tell you which
profiles your deployment can actually serve.

## Where things fit

| Host | GPU | Free | Proposed |
|---|---|---|---|
| vidatron-G192 | 2× RTX PRO 6000 Blackwell 96 GB | **GPU0 idle 96 GB** | planner |
| vegatron-G144 | 3× RTX A6000 48 GB | **GPU0 idle 48 GB** | worker-mtp |
| iridtron-G96 | 1× RTX PRO 6000 96 GB | ~69 GB | gateway + embedder; leave it alone |

Putting the planner on vidatron GPU0 keeps it on the same box as the worker, so
the planner→worker hop inside a turn stays on the PCIe bus rather than the
tailnet, and both share one host-memory pool for HiCache L2.

### One hardware constraint that will bite

**vegatron is Ampere (A6000, sm_86) and cannot run FP8.** The parked
`Qwen3.8-27B` weights are FP8, so `systemctl enable --now qwen38-27b` will bring
up a process that fails on the first forward pass. For the MTP worker on
vegatron you need an int4 checkpoint (AWQ or GPTQ) — 27B at 4-bit is ~14 GB and
leaves plenty of the 48 GB for KV.

If you would rather not maintain a second quant, put the MTP worker on
vidatron GPU0 alongside the planner and skip vegatron entirely: 96 GB holds both
at FP8 if you cap `--mem-fraction-static` per process.

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
