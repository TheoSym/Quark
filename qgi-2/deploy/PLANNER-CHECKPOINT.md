# Choosing the planner checkpoint

Two NVFP4 Flash-Next rebuilds are candidates. Both carry an MTP head, which is
what the spec's `MTP n=2` planner speculation needs, and both target Blackwell —
NVFP4 is a Blackwell format, so neither will run on vegatron's A6000s.

| | `aday777/…-Uncensored-NVFP4-MTP` | `garnermccloud/…-NVFP4-SSD-Stream` |
|---|---|---|
| Size | ~125B total / 6B active MoE | ~180B logical |
| Quant | NVFP4, ModelOpt-compatible RTN, **group 16, no calibration** | NVFP4 W4A4 experts, BF16 attention, FP8 LUT on disk |
| MTP head | yes (one-layer MTP draft experts, NVFP4) | yes, kept under streaming |
| Context | 131K | not stated |
| Host RAM | normal | **frees ~47.6 GiB** — LUT streams from SSD |
| Throughput | not stated | 164.7 tok/s vs 148.5–156.2 for RAM-resident |
| Engine | stock-ish, but needs a build with Qwen4-Exp/Flash-Next + NVFP4 MoE + MTP + PLE CPU-offload | **custom SGLang fork** (`garnermccloud/sglang-ssd-stream`) |

## What actually decides it, for this harness

**Host RAM is not a side note here.** HiCache L2 is sized as a multiple of the
GPU KV pool and lives in host memory, and it is the single biggest lever QGI-2
has on a large stable prefix. SSD-Stream returning ~47.6 GiB of host RAM funds
exactly that. On a box where the planner and worker share one host, that is a
direct trade: LUT in RAM, or prefix cache in RAM.

**Quantization quality lands hardest on the planner.** The planner plans and
answers; the worker only emits schema-constrained JSON, where a small quality
loss is absorbed by the grammar. `aday777`'s RTN with *no calibration dataset*
is the crudest form of PTQ, and it is applied to the model whose output the user
actually reads. That is the wrong place to economise.

**"Uncensored" is a fine-tune, not just a quant.** Whatever the abliteration
does, it is a capability change on top of the quantization, and it is untested
for planning and tool selection. For a coding harness there is no upside.

**A custom engine fork cuts against your own runbook.** `model-serve` is built
around stock vLLM / SGLang / llama.cpp, with a gotcha bank earned on those.
`sglang-ssd-stream` means a second upgrade path, and — the part that matters for
QGI-2 — no guarantee it exposes `sglang:cache_hit_rate` and
`sglang:spec_accept_length`, which are how the harness measures the two metrics
the spec calls bugs when they drop. Verify with `verify-agent-ready.sh` before
committing; it checks for exactly those gauges.

## The risk both share

**An MTP head in the checkpoint is not MTP support in the engine.** `aday777`
says outright that it needs "a recent compatible build with … MTP … support".
If SGLang's NEXTN path does not handle this architecture, the head is inert.

That degrades gracefully rather than failing: set the planner to
`speculation = "off"` in the QGI-2 config and every step still routes — the
planner simply stops speculating. Confirm which you have before reading any
acceptance number, because an unsupported MTP head reports nothing rather than
reporting zero.

## Recommendation

Start with **SSD-Stream on vidatron GPU0**, and treat the fork as the thing to
prove out:

1. `verify-agent-ready.sh` first — it will tell you whether the fork exposes the
   metrics QGI-2 needs. If it does not, that is disqualifying regardless of
   throughput, because the harness would be flying blind on its own premise.
2. Confirm the box has local NVMe. SSD-Stream's numbers assume it; over network
   storage the streaming LUT becomes the bottleneck it was designed to avoid.
3. If either check fails, fall back to `aday777` and accept the quantization
   risk, or serve the planner from the gateway until a better rebuild lands.

## You may not need a new machine

**vidatron GPU0 is idle with 96 GB of Blackwell** — and SSD-Stream was tested on
exactly that card. The planner→worker hop stays on the PCIe bus rather than the
tailnet, which matters because every turn crosses it at least twice.

The argument for a separate box is host RAM and NVMe contention: the 27B worker
already wants host memory for its own HiCache L2, and a streaming LUT wants disk
bandwidth. If `qgi2 hicache --probe` shows the worker's host pool running full
after the planner lands, that is the signal to split them.
