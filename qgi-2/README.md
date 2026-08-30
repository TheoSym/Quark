# QGI-2

An inference-first agent harness implementing [`HARNESS_SPEC.md`](HARNESS_SPEC.md),
built **additively** on jcode.

## The additive guarantee

QGI-2 is a separate cargo workspace that path-depends on jcode's crates.
**No file under `../crates`, `../src`, or `../Cargo.toml` is modified.**
Verify it at any time:

```bash
cd .. && git status --porcelain   # only qgi-2/ should appear
```

jcode is consumed through three extension points it already offers:

| Seam | jcode file | What QGI-2 does with it |
|---|---|---|
| `[providers.<name>]` config | [`jcode-config-types/src/lib.rs:456`](../crates/jcode-config-types/src/lib.rs#L456) | The HTTP edge is reached as an OpenAI-compatible endpoint |
| `Provider` trait | [`jcode-provider-core/src/lib.rs:76`](../crates/jcode-provider-core/src/lib.rs#L76) | The in-process edge implements it |
| `StreamEvent::TokenUsage` | [`jcode-message-types/src/lib.rs:721`](../crates/jcode-message-types/src/lib.rs#L721) | vLLM's `cached_tokens` flows into jcode's existing cache UI |

## Layout

```
crates/
  qgi2-spec-types/     segments, moods, profiles, typed facts, the step triple
  qgi2-factgraph/      the typed graph: store, traversal, deterministic render, retrieval
  qgi2-rules/          compiled Datalog (ascent): verify, tool gating, skills, mood switch
  qgi2-assembler/      cache-shaped assembly + per-segment BLAKE3 hashes
  qgi2-router/         per-step (model, speculation, sampling) + JSON schemas
  qgi2-engine-vllm/    vLLM client: guided decoding, cached_tokens, acceptance scraping
  qgi2-metrics/        the spec's success metrics, reported as defects
  qgi2-turn/           the per-turn loop
  qgi2-edge-http/      OpenAI-compatible sidecar   <- works with stock jcode
  qgi2-edge-provider/  jcode Provider impl         <- needs a variant composition root
bin/qgi2/              the binary
```

## Quick start

```bash
cd qgi-2
cargo build --release

# 1. See the routing table — the spec's "nothing defaults", made visible.
cargo run -p qgi2 -- plan --mood builder --profile traceable

# 2. Write a config, then check your vLLM deployment against it.
cargo run -p qgi2 -- config > ~/.qgi2/config.toml
cargo run -p qgi2 -- doctor

# 3. Serve, and wire stock jcode to it.
cargo run -p qgi2 -- serve
cargo run -p qgi2 -- config --jcode   # prints the [providers.qgi2] block
jcode --provider qgi2
```

The persona rides in the model name — `qgi2/<mood>-<profile>` — so jcode's own
`/model` switcher doubles as a mood switcher, with no QGI-2-specific UI.

## Three things the implementation had to decide

These are places where the spec meets a real constraint. Each is resolved in
code with the reasoning recorded next to it.

### 1. Speculation selects a *process*, not a request field

`--speculative-config` is fixed when a vLLM process starts. There is no
per-request field that switches a running server from DFlash2 to MTP. So
"every step has an explicit `(model, speculation, sampling)` triple" is
satisfied by routing each step to the endpoint launched with that speculation —
[`EngineRegistry`](crates/qgi2-engine-vllm/src/registry.rs).

**Consequence:** the spec's "two processes" is a floor, not a total. Running
Traceable (worker DFlash2 n=7) *and* Deterministic (worker MTP n=3) against one
deployment needs a third vLLM process. A step whose endpoint is not registered
is a hard error — running an MTP-planned step on a DFlash2 server would report
acceptance numbers describing a configuration nobody chose.

`qgi2 doctor` catches this before a session starts.

### 2. Only the sidecar edge reaches *stock* jcode

jcode resolves provider identities through its own name resolution. A
`[providers.qgi2]` block routes to the OpenAI-compatible runtime — that path
needs zero jcode changes and is what `qgi2 serve` targets.

A brand-new provider identity backed by an in-process `Provider` is **not**
reachable via `--provider qgi2` on a stock binary; jcode would have to learn the
name. [`qgi2-edge-provider`](crates/qgi2-edge-provider/) is complete and tested
as a `Provider`, and is used by a variant binary that constructs its own
provider rather than going through jcode's name resolution — which is what the
in-process variant always meant. It is not a change to jcode either way.

### 3. History does not become prompt tokens

Both edges take only the latest user query plus the *trailing* run of tool
results. Replaying the caller's whole transcript would reintroduce exactly the
long, mostly-unchanged prompt the spec exists to eliminate. History still lives
in jcode — transcripts, `/resume`, session search all work — it simply is not
re-sent every turn. What comes back instead is the retrieved subgraph.

Only the *trailing* results are re-read: earlier ones were already extracted on
the round that received them, and re-extracting would double-count facts and
inflate the reinforcement counts that gate promotion to the durable slice.

## Turns and rounds

The spec's per-turn loop assumes the harness runs tools itself. QGI-2 does not —
both edges sit under jcode's agent loop, which executes tools and re-enters the
provider with the results. So one *turn* (one user query) is made of one or more
*rounds*:

```
round 0:  assemble → plan → tool calls ──────────────► jcode executes
round 1:  assemble → extract/verify/commit results
                   → plan → more tool calls ─────────► jcode executes
round 2:  assemble → extract/verify/commit results
                   → plan → answer → extract → verify → commit → mood check
```

The turn index advances only on round 0 and metrics accumulate across rounds, so
"tokens per turn" means what the spec says rather than "tokens per round-trip".
`max_tool_rounds` (default 12) caps the sequence — at the cap the loop forces the
answer step, so a model that keeps asking for tools produces a reply saying the
work was cut short rather than looping forever.

Round state is recovered from the transcript, not tracked server-side. An
OpenAI-compatible endpoint has no session identity — a client may reconnect,
retry, or run several conversations against one server — so a counter kept
beside the conversation would drift from it. The transcript *is* the
conversation.

**Both edges defer execution.** `ToolDisposition::Deferred` is how a runner says
"the caller will run this". An earlier version returned a canned `"deferred"`
string that the loop then fed into fact extraction as though a tool had really
run; the disposition type exists so that cannot happen again.

## How the invariants are enforced

The spec's invariants are load-bearing, so most are enforced by types or tests
rather than by convention:

| Invariant | Enforcement |
|---|---|
| Fixed segment order | `SEGMENT_ORDER` is the only ordering; `SegmentSet` can only render through it |
| Segments 1–3 byte-stable | `Assembler` hashes and reports `PrefixBroken` naming the segment that moved |
| Every structured step under a schema | `StepPlan::validate` rejects a structured step with no schema, and a schema on `Answer` |
| Nothing defaults | `Router::plan` returns `Result`; there is no `Default for StepPlan` |
| Rules validate and commit | `ProposedFact::commit` requires a `CommitToken`, which only `qgi2-rules` can mint |
| Deterministic rendering | Every collection on the render path is a `BTreeMap`/`BTreeSet`; facts sort by identity, never by confidence |
| Speculation never changes the distribution | `Profile::worker_speculation` returns MTP for Deterministic, because DFlash2 cannot do greedy; a test asserts no greedy profile pairs with a non-greedy speculator |
| A cache drop is a bug | `TurnMetrics::breaches` phrases it as a defect and every response carries it |

## Tests

```bash
cargo test --workspace
```

Every crate's tests run without a vLLM deployment; the engine crate is tested
against recorded wire shapes rather than a live server.

## Status

Implemented and unit-tested end to end, including the tool-call path on both
edges — QGI-2 emits calls, jcode executes them, results come back and are
extracted into the graph.

Two things have **not** happened:

- **No run against a real two-process vLLM deployment.** None was available
  here, so the numbers the spec cares about (cache hit rate ≥ 85%, planner
  acceptance ≥ 1.8, worker ≥ 2.0) are unobserved, and the tool loop has never
  executed against a live model. `qgi2 doctor` then `qgi2 plan` are the first
  two things to run once a deployment exists.
- **No content-addressed file memory.** The fact graph stores structure —
  `task:auth depends_on file:auth.rs` — not content. Tool output reaches the
  model verbatim *within* a turn, but across turns only extracted facts survive,
  so the agent re-reads files it has seen before. Cache hit rate will look
  excellent while the agent does redundant work. A `file:x has_digest sha256:…`
  relation would let it know what it has already read and whether it changed;
  that is the next thing worth building for serious repo work.
