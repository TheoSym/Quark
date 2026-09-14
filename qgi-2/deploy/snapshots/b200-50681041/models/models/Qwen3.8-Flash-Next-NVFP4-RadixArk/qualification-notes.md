# Qualification notes

## Conversion

- Source: `Qwen/Qwen3.8-Flash-Next`.
- Path: bounded routed-expert streaming with Model Optimizer snapshot
  `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`.
- Scope: 48 main-model routed-expert layers only (NVFP4 W4A4); MTP and every
  non-routed tensor remain source precision. The PLE n-gram embedding tables
  use the FP8-quantized versions from `Qwen/Qwen3.8-Flash-Next-FP8` (128
  F8_E4M3 shards + per-table scalar scale, declared via
  `text_config.ple_embedding_dtype` in config.json).
- Output: 206 weight shards (192 routed + 4 BF16 + 10 PLE-FP8), 296,475
  tensors, 135,253,624,416 bytes.

## Published checkpoint evidence

- Structural audit: 294,912 routed entries, 128 FP8 PLE entries, 1,433
  unchanged BF16/I64 entries.
- Dtypes: 73,728 U8, 73,856 FP8 E4M3, 147,456 FP32, 1,432 BF16, 3 I64.
- Scale audit: 221,184 finite positive entries, minimum
  2.125331411662046e-05, maximum 448.0.
- Unchanged-content byte-equality audit (routed-expert conversion revision):
  1,562 tensors and 118,408,052,728 bytes compared; all passed, including
  31 MTP tensors.
- Deterministic serving smoke: passed with `finish_reason=stop` and finite
  token logprobs.
- Full-repository verification (verify_hf): PASS on commit `3818f20f7c7c`,
  420 files, all metadata and LFS object hashes matched.

## GSM8K (this checkpoint)

- Protocol: full 1319, single-shot, sgl-eval @ 645cf56, temperature 0.6,
  top-p 0.95, max_tokens 8192, threads 32, seed 0.
- Result (current revision, FP8 PLE): score 97.27% (1283/1319), stop_rate
  98.86%, error_rate 0.0%, truncated_rate 1.14%.
- Earlier result (BF16-PLE revision): score 97.35% (1284/1319), stop_rate
  99.47%, error_rate 0.0%, truncated_rate 0.53%.
- Verdict: accuracy inside the BF16 reference band (97.12-97.50); stop-rate
  98.86% is 0.14pp below the round-1 frozen 99% line and is recorded as-is
  (accepted by the reviewer).
- Raw metrics: `gsm8k_metrics.json`.

## AIME26 (this checkpoint)

- Protocol: 30 problems x 8 repeats, sgl-eval @ 645cf56, temperature 1.0,
  max_tokens 130000, threads 8, seed 0.
- Result: pass@1 98.75% (237/240, SEM 0.61pp), majority@8 100%, pass@8 100%,
  stop_rate 99.17%, truncated 0.83%, no_answer 0.42%, error 0.0%. Measured
  on the BF16-PLE revision; the two revisions differ only in the PLE tables.
- Verdict: inside the BF16 reference band (>= 236/240); the BF16 port run on
  the same source scored 100% (240/240).
- Raw metrics: `aime26_metrics.json`.

## Serving requirements

- Runtime: SGLang with `qwen4_exp` support plus one of: (a) the fp8-resident
  PLE loader in `Qiaolin-Yu/sglang-qwen-next` PR #40 (uses the declared
  `ple_embedding_dtype`), or (b) a loader that dequantizes the FP8 PLE table
  with its scalar `weight_scale` at load time. Loaders that only upcast the
  FP8 bytes will serve wrong PLE embeddings silently.
- Recommended sampling (from generation_config): temperature 1.0, top-p
  0.95, top-k 20.

## Qualification boundary

An earlier independently converted candidate from the same source scored
AIME26 97.92% (235/240). Because activation capture was repeated for this
published checkpoint, the two sets of quantized weights are not assumed
byte-equal and that earlier score is not attributed here.
