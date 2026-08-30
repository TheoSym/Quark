# Inference-First Agent Harness — Abstract Spec (v2, jcode-based)

The prior version put the harness first and inference second. This version inverts that: **the inference layer is the product; the harness exists to keep the inference layer fast.** Details and choices live in `HARNESS_DETAILED.md`.

## Thesis

Agent loops are slow and inaccurate because every turn re-sends a long, mostly-unchanged prompt and lets the model manage its own memory. Fix both at the inference layer:

1. **Make the prompt cache-shaped.** Fixed segment order, stable bytes at the front, volatile bytes at the end. The KV/recurrent-state cache does the rest.  
2. **Make output predictable.** Constrained decoding for every structured step, so speculation accepts more and rules can trust the result.  
3. **Speculate everywhere.** MTP on the planner, DFlash2 or MTP on the worker, n-gram where the output copies the prompt.  
4. **Keep memory small and typed.** A fact graph in RAM, rendered deterministically, so the volatile tail stays short.  
5. **Two models, one control layer.** A big planner and a small worker, each with its own speculation and cache, selected per step by the harness.

## Components

| Layer | Choice | Role |
| :---- | :---- | :---- |
| Engine | vLLM, two processes | Prefix caching, speculation, structured outputs, OpenAI-compatible API |
| Planner | Qwen3.8-Flash-Next (NVFP4) | Plan and answer. GDN \+ QSA hybrid attention, PLE n-gram embedding, MTP head, 262K native context |
| Worker | Qwen3.8-27B (NVFP4) \+ DFlash2 or MTP | Extract, render, route, tool-args, cascade first-try |
| KV persistence | LMCache / vLLM KV connector | Cached prefix survives restarts and is shared across replicas |
| Control layer | Harness-owned router \+ assembler | Per-step model/spec/sampling selection; cache-aware prompt assembly with segment hashes |
| Memory | jcode's in-RAM graph, extended with typed facts | `(subject, relation, object, confidence, source, turn)` on each entry |
| Rules | Compiled Datalog in Rust (`crepe`/`ascent`) | Retrieval traversal, tool gating, consistency, skill selection, mood switching |
| Embedder | Qwen3-Embedding-0.6B via vLLM `/v1/embeddings` (MiniLM fallback) | Entry-point retrieval only |
| Outer loop | jcode (Rust) | TUI, tools, stdio MCP, swarm, lazy skills, session resume, cache-miss warnings |

## Invariants

- Prompt order is always `core → mood → durable slice → active skills → session subgraph → query`. Segments 1–3 are byte-stable within a session; 4–6 are the only recomputed tokens.  
- Every structured step (extract, render, tool-args, route) runs under a JSON schema. No free-text bookkeeping.  
- Every step has an explicit `(model, speculation, sampling)` triple chosen by the router from mood and profile. Nothing defaults.  
- Cache hit rate is measured every turn from vLLM's `cached_tokens` and surfaced in the UI. A drop below threshold is a bug, not a metric.  
- The model proposes facts; rules validate and commit. The graph is never written by the model directly.  
- Rendering is deterministic: same graph → same bytes → same cache blocks.  
- Speculation never changes output distribution; accuracy comes from constrained decoding, typed memory, and rules, not from speed features.

## Per-turn loop

```
assemble(core, mood, durable, skills, subgraph, query)   # hashes per segment, cache check
→ plan        [planner, MTP n=2, thinking per profile]
→ tool calls  [args under schema; mask from rules]
→ extract     [worker, DFlash2 n=7 | MTP n=3, schema]
→ verify      [rules: dedupe, conflict, confidence floor]
→ commit      [graph write, derived views refresh]
→ answer      [planner]
→ extract answer facts → verify → commit
→ mood check  [rules]
```

Session end: promote to durable, decay, log speculation acceptance and cache-hit stats as facts for the self-tuning loop.

## Moods (config over one core)

|  | Builder | Researcher | Companion |
| :---- | :---- | :---- | :---- |
| Traversal | Task→depends\_on→File | Claim→supports/contradicts→Source | Person→prefers→Topic |
| Conflict | latest wins | keep both | highest confidence |
| Tools | fs, shell, git | web, fetch, docs | calendar, mail, notes |
| Planner sampling | T 0.3 | T 0.7 | T 0.7 |
| Worker spec | DFlash2 | DFlash2 | DFlash2 |

## Profiles (orthogonal to moods)

|  | Traceable | Deterministic | Quick |
| :---- | :---- | :---- | :---- |
| Worker spec | DFlash2 n=7 | **MTP n=3** (DFlash2 can't do greedy) | DFlash2 n=7, n-gram fallback |
| Sampling | as mood | T 0, seed fixed, batch-invariant | as mood, thinking off |
| Memory sync | async (jcode default) | **sync** (await side-agent) | async |
| Retrieval | full chain \+ reranker | full chain, turn-based decay | exact-key \+ lexical, BFS depth 1 |
| Logging | prompts, segment hashes, rule firings, acceptance rates | hashes \+ seeds \+ engine build | errors only |

## Success metrics

- Prefix-cache hit rate per model ≥ 85% (from `cached_tokens`).  
- Speculation acceptance: planner MTP ≥ 1.8 tokens/step; worker DFlash2 ≥ 2.0.  
- Tokens per turn trending down as memory replaces raw context.  
- Extraction rejection rate at verify ≤ 10%.  
- Planner:worker token ratio ≤ 1:3.

## Non-goals

- Writing an inference engine. vLLM is the engine; the harness owns the control layer above it.  
- Cross-model KV sharing. Not production-ready.  
- Same model twice. No capability gain; only replicas for throughput.

