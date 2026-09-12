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

- **SSH:** `ssh -i ~/.ssh/ai-infra -p 32803 root@<old-instance-public-ip>` (direct port;
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
  Caddy token edge as external port 10100 → `http://<old-instance-public-ip>:32819/v1`,
  auth `Authorization: Bearer $OPEN_BUTTON_TOKEN` (the instance's portal token;
  put it in the qgi2 config's `api_key`). For a private path with no token,
  SSH-forward instead: `ssh -i ~/.ssh/ai-infra -p 32803 -L 18034:127.0.0.1:18034 root@<old-instance-public-ip>`
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

### First live run — 4× B200, vast instance 50681041 (2026-09-12)

The RTX PRO 6000 box was lost to a failed VM conversion; the replacement is
4× B200 183 GB, 2 TB host RAM, 59 Gbps down (weights land at ~5 GB/s). All
four GPUs initialise together. Same recipe: SGLang from commit `824bb45e` in
`/venv/sglang`, weights verified at `/models`, supervisor program `qgi2-v41`,
Caddy on external port 10100 → `http://<instance-public-ip>:37972/v1` with the
instance token.

What the bringup taught, now in the scripts:

- `--language-model-only` is rejected for `DeepseekV4ForCausalLM` at this
  commit; the class is text-only anyway. Removed.
- The `dsv4` attention backend fixes `page_size` at 256; the config's
  `[hicache] page_size` says 256 to match.
- First launch JIT-compiles FlashInfer's SM100 kernels and runs autotune:
  about 35 minutes, once. Later launches reuse `/root/.cache/sglang`.
- `reasoning_effort` on the wire is the OpenAI literal
  (`none|minimal|low|medium|high|xhigh|max`), not the model card's 1–100
  integer; the harness maps its numeric table at the wire.
- `--enable-cache-report` is required or `cached_tokens` is 0 on every reply
  and the harness flags a false cache breach. The server-side
  `sglang:cache_hit_rate` gauge showed 0.67 with 2,560 cached tokens on the
  same run, so the prefix was being reused all along.
- `verify-agent-ready.sh` now accepts `tool_calls` on `/v1/chat/completions`
  when `/v1/responses` returns text; chat completions is the path both QGI-2
  and jcode use, and it produced a correct tool call on the first try.

Measured on a fresh three-turn session, no tools, after cache reporting was
enabled and the page-aware cache floor landed:

| | turn 1 | turn 2 | turn 3 |
|---|---|---|---|
| wall time | 2.3 s | 6.1 s | 4.4 s |
| prompt tokens (all steps) | 2087 | 2032 | 2051 |
| cached tokens | 1536 (74 %) | 1536 (76 %) | 1536 (75 %) |
| breaches | none | none | none |
| facts committed | 0 | 0 | 2 |
| verify rejection | 0 % | 0 % | 0 % |

Reading the cache number: every harness request here is three 256-token pages
and the engine caches exactly the two full pages before the page that holds
the volatile tail, so 67–76 % *is* the ceiling for prompts this short at this
page size. The harness now bounds its 85 % floor by that ceiling (see
`TurnMetrics::cache_hit_ceiling`) and reports a breach only when the prefix
does worse than the page size allows. Turn 1 of a new session already hits,
because the engine's radix cache still holds the persona's prefix from earlier
sessions. The ratio rises on its own as the durable slice and subgraph grow.

Engine gauges over the run: DSpark acceptance 2.6–2.8 tokens/step on mixed
harness traffic and 3.9 on a plain coding prompt, all above the spec's 1.8
planner floor; single-stream decode ~267 tok/s at `reasoning_effort: low`.
One earlier turn rejected 2 of 3 proposed facts at verify; later turns
rejected none. Watch that rate on real coding sessions.

### Two cards are enough (2026-09-12)

The four-card layout was mostly empty space: 73 GB of weights per card and a
KV pool auto-sized to 39 million tokens, plus 556 GB of host RAM pinned for a
HiCache tier that could never get a hit because the GPU pool never evicted.
Measured alternatives:

| Layout | Weights + draft per card | KV pool | Per card used | Host RSS | Result |
|---|---|---|---|---|---|
| TP4, fraction 0.8, HiCache ratio 2 | 75 GB | 39.5 M tokens | 157 GB | 740 GB | serves; tier idle |
| TP3 | | | | | refused: vocab 129,280 is not divisible by 3 |
| TP2, fraction 0.8 | 147 GB | | | | refused: needs ≥ 0.831 for the weights |
| **TP2, fraction 0.92, pool capped at 4 M tokens, HiCache off** | 147 GB | 4 M tokens | 166 GB | 199 GB | **serves; turns 3–10 s; DSpark 2.9** |

The two-card layout is the standing configuration in the service wrapper
(`CUDA_VISIBLE_DEVICES=0,1 TP=2 MEM_FRAC=0.92 MAX_TOTAL_TOKENS=4000000`),
which frees cards 2–3 for a second replica or the 27B worker. HiCache is
turned back on for this layout, where the capped pool makes eviction real and
the ratio-2 host tier costs tens of GB rather than hundreds. Concurrency per
process: `--max-running-requests 32`; DSpark's advantage holds to about 4
streams and fades past 8, so plan ~8 interactive users per process.

### What the first memory session found in the harness (2026-09-12)

A three-turn session that states two facts about a file and then asks what
is known about it. Before the fixes below the third turn answered "I have no
facts about src/auth.rs" with two committed; after them it answers from the
graph, in 2–2.5 s per turn, zero rejections, no breaches.

- **Structural facts were committed but never rendered.** Verify accepts
  `is_a`/`part_of` in every mood; the walk only rendered the mood's traversal
  relations. Entry points now render their structural facts too; expansion
  still follows the mood only, and off-mood relations stay invisible.
- **The route step could name a node that does not exist**, and a walk from
  it reached nothing. Its entry points are now filtered to nodes in the
  graph, falling back to the retrieval candidates.
- **The extract schema offered every relation**, so the builder extractor
  proposed `cited_by` and verify threw it away. The enum is now the mood's
  traversal relations plus the structural ones — exactly what verify accepts.
- **The answer step leaked raw tool-call markup** (`<｜DSML｜ calls>`) when
  the plan had asked for a tool the caller did not offer: the answer request
  carries no tool definitions, so the engine's parser does not run on it. The
  answer prompt now says no tool can be called from that step; the response
  flags `answer_contained_tool_markup` if it happens anyway.
- **Duplicates counted as rejections.** The subgraph shows the model its own
  facts and it re-proposes them; that is wasted worker tokens, not wrong
  extraction. `rejection_rate` now excludes duplicates and reports them
  separately, and the extract prompt asks for facts that are new and spells
  out the direction of `subject relation object`.
- **The cache floor asked for too much.** Steps in a turn share the stable
  prefix but append their own tails, and once a tail crossed a page boundary
  the "all pages but the last" ceiling flagged a byte-identical prefix. The
  floor is now the stable prefix's full pages — what the harness controls.
- Verify rejections and the retrieval summary (entries, unmatched, reached)
  are logged per turn under the Traceable profile.

### First real coding runs through stock jcode (2026-09-12)

`jcode run --json --provider-profile qgi2 --model qgi2/builder-traceable@<session>`
against the bench tasks, with jcode's own 34 tools as the runner. The first
A/B pass (`bench/ab_compare.py`, direct engine vs harness, same model) was
**baseline 4/4, harness 1/4** — and every harness failure was a harness
defect, found with the new per-round trace (`plan`, `tool call`, `tool
result`, `tool mask denied`). Fixed:

- jcode wants `--provider-profile`, not `--provider`, for a `[providers.X]`
  block; its `--json` is one document, not NDJSON. The bench handles both.
- jcode sends `stream: true`; the edge answered with a JSON body and jcode
  saw an empty response. The edge now streams standard chunks.
- The planner was never told the exact tool names and invented `fs.read`.
  The plan prompt lists the caller's tools — only those the mood mask admits,
  since listing `batch` and then refusing it made the planner give up.
- `needs_tools: false` beside a `bash:` step: any step naming a tool is a
  tool request now. A step whose intent starts with a tool name but names no
  tool (`{"intent": "read app/models.py"}`) is taken as that tool.
- The tool-args worker saw only the intent; it filled `agentgrep` with a
  plausible mix of fields the tool rejected. It now sees the tool's
  description and memory retrieved by the step's own intent.
- Every jcode run on a persona shared one harness session, so one task's
  `calc.py` facts leaked into the next task's `calc.py`. `@client` in the
  model name is the per-run session channel.
- Tool output stored in the line store was sentence-split and
  whitespace-collapsed, so the `edit` after a `read` in the next round could
  not match bytes and the planner re-read the same file five rounds running.
  Tool output is now a verbatim block, and the previous round's results ride
  in the prompt as a working window.

Second A/B pass after the fixes, two repeats, same model on both arms:

| arm | solved | follow-up turn | tokens (8 runs) | cache | median wall |
|---|---|---|---|---|---|
| stock jcode, direct to engine | 8/8 | 100 % | 170,351 | 98 % | 5.8 s |
| stock jcode through QGI-2 | 6/8 | 100 % | 218,945 | 50 % | 33.8 s |

The two harness misses: one run stopped to ask "tell me which" instead of
acting, and one `edit` missed the file's whitespace byte-for-byte. The
memory-dependent task (`stateful`) passed on both arms both times.

Reading it honestly: on this model the plan → tool-args → extract split costs
two to three engine requests per round where stock jcode spends one, so the
harness is slower and spends *more* tokens, and the split-off planner is a
worse tool user than the model calling tools natively. The harness's value
today is memory, metrics and the cache discipline, not speed. The next
design step is to let the plan step call tools natively (the engine's parser
already returns structured `tool_calls`), keep the schema on extract, and
skip route/extract on rounds with nothing to route or extract.

### Memory-only mode (2026-09-12, evening)

The decision after the second A/B: jcode stays native for everything, and
QGI-2 does only what the Q-hypergraph note measured — extraction into the
line store and compaction on events. `qgi2 serve --proxy [--compact]` is a
pass-through in front of the planner endpoint that

- records every message of the transcript into the session's line store
  once (user turns and answers by sentence, tool output as a verbatim
  block; jcode's `<system-reminder>` messages skipped), with no model call;
- retrieves the lines that matter for the turn's query and inserts them as
  one `<memory>` message just before the latest user message, frozen for
  the turn so the prefix jcode sends stays byte-identical across rounds;
- when the budget is hit and `--compact` is on, makes one engine call under
  a JSON schema to choose which unjudged candidates to keep; the verdicts
  persist and a line is never re-judged;
- forwards everything else untouched, streaming included, and reports
  `x-qgi2-memory-lines` and `x-qgi2-memory-budget-hit` on the response.

Sessions are selected with `--model QGI-2-V41@<session>` (the suffix is
stripped before forwarding) and persist under `[server] graph_dir`. jcode
profile: `[providers.qgi2mem]` at `http://127.0.0.1:8789/v1`.

Third A/B pass, memory-only proxy as the QGI-2 arm, two repeats:

| arm | solved | follow-up turn | tokens (8 runs) | cache | median wall |
|---|---|---|---|---|---|
| stock jcode, direct | 8/8 | 100 % | 168,092 | 98 % | 5.5 s |
| stock jcode through the memory proxy | 8/8 | 100 % | 174,168 | 91 % | 5.4 s |

Parity on every task at the same speed; the 4 % token overhead is the
`<memory>` message, and the cache figure drops only because that message
sits after the cached prefix. No compaction event fired: these tasks never
approach the 30-line budget. The memory advantage is a long-session
property — the note's 124K-token transcripts — and the next measurement is a
multi-hour session replay, not this bench.

Line-store memory (`qgi2-memory`, the Q-hypergraph port) is wired in:
queries, answers and tool output enter the store; deterministic retrieval by
term overlap, anchor expansion, per-turn coverage and neighbours renders into
segment 5 under a 30-line / 6 KB budget; `memory_budget_hit` on the response
is the compaction counter. No vector index yet: none of the fleet's embedders
is configured, and zvec has no Rust binding — a vector path, when wanted,
goes through the existing embedder slot with an in-process ANN crate.

### The prototype's ingester, incremental compaction, and the code graph (2026-09-12, night)

The first port of the memory approximated the Q-hypergraph ingester with a
lexicon scan and no service; that was rejected. The proxy now runs the
prototype's pieces as they are:

- **Deterministic ingester** (`qgi2-memory/src/parse.rs`, from
  `QHP-extraction/q-hypergraph/qhg_extract.py`). Prose is split by the
  sym-tools spaCy service (`/sentences`) and parsed in batches of 48
  (`/batch-parse`: subject | relation | object, negation, modal) behind a
  content-addressed per-sentence parse cache (`graph_dir/parse-cache.json`).
  The modality map, the constituency pattern and sym's role assigner in its
  original priority order run on that parse; the predicate, the assigner's
  confidence and the signal that fired are persisted on every line, and the
  utterance's negation pairs and discourse links on the store. Code fences,
  tables and tracebacks stay whole as `Block` lines; tool output stays one
  verbatim block. If the service is down the lexicon path is used and each
  such line carries `signal = "lexicon_fallback"`. The service runs locally
  (`QHP-CORE-1/services/sym-tools`, `uvicorn main:app --port 8100`, spaCy
  3.8.14 + en_core_web_sm); `qgi2 serve --proxy --sym-tools <url>` points
  at it (default `http://127.0.0.1:8100`, `off` disables).
- **Incremental compaction** (`--compact every-turn`, policy B of
  `phase_g/tier2/incremental_select.py`): every turn the compactor sees only
  the candidates it has never judged, the kept set persists, and coverage
  picks (best lines of every hitting turn) bypass the verdict as the
  prototype's `kept | coverage_hits` does. `--compact` alone keeps the
  events-only behaviour; `off` counts budget hits. The memory message now
  opens with the per-turn entity registry (`session_headers`: the
  capitalised terms the user used in each represented turn).
- **Code graph MCP.** `codebase-memory-mcp` v0.10.8 is registered in
  `~/.jcode/mcp.json` as `codebase_memory` (`shared = true`); the QGI-2
  workspace is indexed (2,285 nodes, 11,060 edges). jcode exposes it
  natively as `mcp__codebase_memory__*`; nothing in the proxy knows about it.

Verified live against the B200 planner: a jcode run through the proxy
answered "where is `add_parsed` defined" with the MCP `search_graph` call
only; the store for that session holds the parsed predicates (e.g. `I |
search | the code graph`, modal `'ll`, role Action), the compactor judged 2
then 4 new candidates on turns 1 and 2, and a second, separate jcode process
in the same memory session recalled the file and line range with tools
forbidden, from the `<memory>` message alone.

Fourth A/B pass, same bench, proxy with the spaCy ingester and
`--compact every-turn` (one compaction call per turn, on new candidates):

| arm | solved | tokens (8 runs) | cache | median wall |
|---|---|---|---|---|
| stock jcode, direct | 8/8 | 235,727 | 99 % | 6.6 s |
| stock jcode through the memory proxy | 8/8 | 236,845 | 96 % | 7.7 s |

Parity on solve rate and tokens; the extra second of wall clock is the
per-turn compaction call plus the parse round-trips, which `--compact`
(events only) removes on short sessions.

**Cost note:** the B200 instance bills about $28/h whether or not it serves.
Stop it between sessions (`vastai stop instance 50681041`); the weights,
venv and JIT caches survive a stop/start, not a recycle.

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
