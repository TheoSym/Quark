# QHG in the Harness — Two Modes, One Week (spec)

Companion to `HARNESS_SPEC.md`. Goal: a realistic QHG implementation inside the inference harness in **under a week**, in two modes. No Q-GST L1/L2/L3, no new storage platform — the fact graph already in `qgi2-factgraph` is the substrate; QHP contributes the **language** (QLang), the **semantics** (roles, modality, derived relations), and an **offline front-end** (`sym-ingest`).

## Thesis

The harness spec already demands exactly what QHG provides:

| Harness spec says | QHP already has |
| :---- | :---- |
| "typed facts `(subject, relation, object, confidence, source, turn)`" | `SymPredicate { subject, relation, object, negated, modality, confidence }` |
| "rendered deterministically, so the volatile tail stays short" | QLang: deterministic role-prefixed sentences, designed to be "converted back into graphs, rules, or logic" |
| "Rules validate and commit… retrieval traversal, tool gating, consistency" | QHG derived relations (`implies`, `conflicts_with`, `depends_on`) + deontic modality (obligation/prohibition/permission) |
| "The model proposes facts; rules validate" | QHG's core decision: the hypergraph is a **working structure built at query time; only its derived outputs persist** |

So neither mode invents anything. Mode 1 is a *renderer*; Mode 2 is a *pre-pass*.

## Mode 1 — Rules as sentences: QLang as compacted LLM memory

Facts render not as terse triples (`claim:x supports source:y [0.80]`) but as **numbered, role-prefixed QLang sentences** — the format QHP's pipeline produces and every LLM reads natively:

```
1. Obligation: The borrower must repay the loan within 30 days.
2. Cause: A missed payment triggers the delinquency process.
3. Condition: If the credit score falls below 600, then manual review applies.
```

- **Renderer** `render_qlang()` beside `render_facts()` in `qgi2-factgraph::render`. Role + sentence derived **deterministically** from `(relation, modality)` via fixed CNL templates (QHP `Ingest-QHG.reference` §6.6: `Obligation → "{subject} MUST {relation} {object}"`, etc.). Same identity sort, same budget, same byte-stability invariant — no timestamps, no LLM in the render path.
- **Modality** on facts: `fact | obligation | prohibition | permission | recommendation` (QHP's `SymModality`). Lives in fact metadata, NOT in `FactKey` — identity stays `(subject, relation, object)`.
- **Compaction**: when history is evicted/summarized, the per-turn extract step (already schema-constrained on the worker) is the compactor — evicted turns survive as QLang sentences in the durable slice. One sentence ≈ 12–20 tokens vs a paragraph of dialogue.
- **Import**: a loader from `sym-ingest` output (`rules/*.json`: `{ rule_id, rule_type, text, predicate{subject,relation,object}, provenance }`) → `ProposedFact`s. This is the "provide the graphs or rules as sentences" entry: run any document through `sym-ingest` (zero LLM, deterministic), load the rules, and the agent starts with that document as durable memory.

## Mode 2 — Hypergraph pre-reasoning: helping the model concentrate

Before the planner sees the prompt, a rules pre-pass builds QHG's **working hypergraph** over the retrieved candidate facts and renders a short **Focus block** at the tail of the session-subgraph segment:

```
FOCUS
Conflicts: (2) claim:rate-cut contradicts source:fed-minutes; …
Obligations active: borrower must-repay loan-2201 (unmet)
Goal chain: task:close-loan ← depends_on ← task:verify-income ← [MISSING]
```

- **Pre-pass** in `qgi2-rules` (Datalog, `crepe`/`ascent` per spec): over candidates only —
  1. **Conflicts** — `Relation::negation()` pairs plus modality clash (obligation vs prohibition on same subject/object).
  2. **Dependency closure** — transitive `depends_on` from the current goal; report the first unmet link.
  3. **Obligation activation** — deontic facts whose condition-facts are present ⇒ "active"; these also feed the existing tool mask (a `prohibition` on an action gates the tool).
  4. **Implications** — `implies` derivation where rule chains compose.
- **Hyperedges as groupings**: a query-time overlay `Hyperedge { connects: Vec<FactId>, kind }` for process-shaped groups (sequential steps, condition→action) — QHP's process hyperedges. **Never persisted.** Only derived findings commit back, as facts with `Source::Rule` (already trusted, skips verify) — the harness-scale version of `QHG_Derived_Relation`.
- **Cache shape**: the Focus block is deterministic (identity-sorted, fixed templates, no counts of turns/time) and sits at the end of segment 5, so segments 1–4 stay byte-stable. Invariant unchanged: `core → mood → durable → skills → subgraph(+focus) → query`.

## Week plan

| Day | Deliverable |
| :---- | :---- |
| 1 | `qgi2-spec-types`: `Modality`, role derivation table, CNL templates; byte-stability tests |
| 2 | `qgi2-factgraph`: `render_qlang()` + Focus-block renderer; `sym-ingest` rules loader |
| 3 | `qgi2-rules`: Datalog pre-pass (conflicts, dep-closure, obligation activation) → `DerivedFinding` → trusted commit |
| 4 | Hyperedge overlay; assembler wiring; tool-mask hookup from prohibitions |
| 5 | End-to-end: `sym-ingest` a real doc → load → agentic session; measure tokens/turn + cache-hit before/after |
| 6–7 | Buffer: QLang↔fact round-trip eval (reuse QHP-Research 35-sentence gold set), Deterministic-profile check (T0, fixed seed, byte-identical prompts) |

## Success metrics (added to the harness set)

- Durable slice tokens for an ingested document ≤ 25% of the raw document tokens at equal task accuracy.
- Focus block ≤ 512 bytes rendered, and prefix-cache hit stays ≥ 85%.
- Pre-pass wall time ≤ 5 ms per turn at 10k facts (in-RAM Datalog; if not, cut candidate set, not the rules).
- Round-trip: fact → QLang sentence → worker extract → same `FactKey` ≥ 95% on the gold set.

## Non-goals (this week)

- Q-GST L1/L2/L3, SpacetimeDB, Qvec — the 2026-08-31 direction shelved the big stack; this IS the "speed and usage" path.
- semantica / QGI-QHG-platform integration — parallel platform track, not this harness.
- Persistent n-ary hyperedge storage — hyperedges stay query-time overlays; findings persist as plain facts.
- New extraction models. `sym-ingest` (offline) and the existing worker extract step (online) are the only extractors.

## QHP sources of truth

- Role vocabulary + format rules: `QHP-CORE/core/ingest/prompts/qlang.ts`
- CNL templates + modality/entity vocabulary: `QHP-CORE-1/Ingest-QHG.reference` §6
- Derived-relation semantics + "working structure, not storage": `QHP-extraction/HYPERGRAPH.md` §3
- Sym rule JSON shape: `QHP-CORE/core/sym/writer.ts` (`SymRuleEntry`)
- Gold set for round-trip eval: `QHP-extraction/06-2026-04-11_atomic-tool-benchmark/`
