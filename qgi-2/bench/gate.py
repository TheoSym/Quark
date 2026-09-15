#!/usr/bin/env python
"""The gate: one command, one model, about fifteen minutes, one verdict line.

Four legs, each a fixed slice of a bench that already exists in this
directory, so a gate result compares directly with the full runs:

  engine    readiness (a tool call comes back structured), TTFT on a ~1.5K
            prompt, single-stream decode tok/s, whether the endpoint reports
            cached tokens                                          ~1 min
  coding    polyglot_bench.py, 3 Exercism exercises, jcode          3-8 min
  agentic   docbench.py, 2 document tasks (the fastest verifiers) +
            ab_compare.py `stateful` (a follow-up turn that depends on
            the previous one)                                       3-6 min
  memory    long_horizon.py, direct-full, 40 turns / 8 facts: does the
            model recall planted facts from a 40-turn context, and
            what does that context cost                            2-3 min

Correctness is the sum of objective verdicts over the legs (test suites,
verifiers, exact-value recall); nothing is LLM-judged. Speed is the coding
leg's median wall per task plus the engine leg's decode rate and TTFT.

    python gate.py --model "Gemini 3.8 Flash" --jcode-profile qgillm --bench <polyglot dir>
    python gate.py --report results/gate          # table over every gate-*.json

A gate is a gate, not a ranking: three exercises separate "works" from
"does not", not a 90 % model from a 93 % one. Speed is only comparable
across runs on the same endpoint at the same concurrency; the JSON records
the endpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_JCODE = REPO / "target" / "selfdev" / ("jcode.exe" if os.name == "nt" else "jcode")

CODING_EXERCISES = ["bowling", "forth", "wordy"]
DOC_TASKS = ["pdf-invoice", "xlsx-model"]
AB_TASKS = ["stateful"]
LH_TURNS, LH_FACTS = 40, 8


# ---------------------------------------------------------------- endpoint
def jcode_provider(profile: str) -> tuple[str | None, str | None]:
    """base_url and api_key_env for a [providers.<profile>] block in ~/.jcode/config.toml."""
    cfg = Path.home() / ".jcode" / "config.toml"
    if not cfg.exists():
        return None, None
    text = cfg.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^\[providers\.{re.escape(profile)}\]\s*$(.*?)(?=^\[|\Z)", text, re.S | re.M)
    if not m:
        return None, None
    block = m.group(1)
    base = re.search(r'^base_url\s*=\s*"([^"]+)"', block, re.M)
    env = re.search(r'^api_key_env\s*=\s*"([^"]+)"', block, re.M)
    env_file = re.search(r'^env_file\s*=\s*"([^"]+)"', block, re.M)
    key = None
    if env:
        key = os.environ.get(env.group(1))
        if not key and env_file:
            f = Path.home() / ".jcode" / env_file.group(1)
            if f.exists():
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith(env.group(1) + "="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return (base.group(1) if base else None), key


def jcode_key_env(profile: str) -> str | None:
    cfg = Path.home() / ".jcode" / "config.toml"
    if not cfg.exists():
        return None
    text = cfg.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^\[providers\.{re.escape(profile)}\]\s*$(.*?)(?=^\[|\Z)", text, re.S | re.M)
    env = m and re.search(r'^api_key_env\s*=\s*"([^"]+)"', m.group(1), re.M)
    return env.group(1) if env else None


def repo_env(name: str) -> str | None:
    """A value from the repo's .env (gitignored), for the direct-API legs when
    the jcode profile keeps its key somewhere this script cannot see."""
    f = REPO / ".env"
    if not f.exists():
        return None
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def chat(base: str, key: str | None, body: dict, timeout: int = 300) -> tuple[dict, float]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps({**EXTRA_BODY, **body}).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def ttft(base: str, key: str | None, body: dict, timeout: int = 300) -> float | None:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps({**EXTRA_BODY, **body, "stream": True}).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                try:
                    d = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    return time.time() - t0
    return None


LONG_PROMPT = ("def f(x):\n    return x * 2\n" * 250) + "\nIn one sentence, what does the code above do?"

# Request fields every direct-API call carries (reasoning_effort, usage.include);
# set once in main from the CLI.
EXTRA_BODY: dict = {}


def openrouter_pricing(model: str, key: str | None) -> dict | None:
    """USD per token for a model on OpenRouter: prompt, completion, cache read."""
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {key}"} if key else {})
        data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    except Exception as e:  # noqa: BLE001
        print(f"   pricing unavailable: {str(e)[:120]}", flush=True)
        return None
    for m in data:
        if m["id"] == model:
            p = m["pricing"]
            return {"prompt": float(p.get("prompt") or 0), "completion": float(p.get("completion") or 0),
                    "cache_read": float(p.get("input_cache_read") or p.get("prompt") or 0),
                    "context_length": m.get("context_length")}
    return None


def openrouter_key_usage(key: str | None) -> float | None:
    """Total USD spent on this key so far; the delta across a gate is the exact
    cost, reasoning tokens included, which jcode's token counts leave out."""
    if not key:
        return None
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"})
        return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
    except Exception:  # noqa: BLE001
        return None


def cost_of(pricing: dict | None, input_tokens: int | None, output_tokens: int | None, cached: int | None) -> float | None:
    """jcode reports input (inclusive of cache reads), output, and cache-read tokens."""
    if not pricing or input_tokens is None:
        return None
    i, o, c = input_tokens or 0, output_tokens or 0, cached or 0
    return round((i - c) * pricing["prompt"] + c * pricing["cache_read"] + o * pricing["completion"], 4)


def leg_engine(base: str, key: str | None, model: str) -> dict:
    out: dict = {"base": redact(base), "errors": [], "cost_usd": 0.0}
    t_leg = time.time()
    # 1. readiness: a tool call must come back as a structured tool_calls entry.
    try:
        d, _ = chat(base, key, {
            # 1024: reasoning tokens count against max_tokens on the gateway, and
            # a reasoner that spends 200 thinking would read as "no tool call".
            "model": model, "max_tokens": 1024, "temperature": 0,
            "messages": [{"role": "user", "content": "Use the read_file tool to open README.md. Call the tool; do not answer in prose."}],
            "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
            "tool_choice": "auto",
        })
        msg = d["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        out["tool_call"] = bool(calls) and calls[0].get("function", {}).get("name") == "read_file"
        out["tool_call_raw_in_content"] = (not calls) and ("read_file" in (msg.get("content") or ""))
        out["cost_usd"] += (d.get("usage") or {}).get("cost") or 0
    except Exception as e:  # noqa: BLE001
        out["tool_call"] = False
        out["errors"].append(f"readiness: {str(e)[:160]}")
    # 2. TTFT on a ~1.5K-token prompt, twice (second shows any prefix cache), keep the median of 2 fresh + note the repeat.
    try:
        t = []
        for i in range(2):
            t.append(ttft(base, key, {"model": model, "max_tokens": 32, "temperature": 0,
                                      "messages": [{"role": "user", "content": LONG_PROMPT + f" (run {i} {time.time()})"}]}))
        out["ttft_s"] = round(min(x for x in t if x is not None), 2) if any(t) else None
    except Exception as e:  # noqa: BLE001
        out["ttft_s"] = None
        out["errors"].append(f"ttft: {str(e)[:160]}")
    # 3. decode rate: one 300-token completion, twice, median.
    try:
        rates = []
        for i in range(2):
            d, s = chat(base, key, {"model": model, "max_tokens": 300, "temperature": 0,
                                    "messages": [{"role": "user", "content": f"Write a detailed 300-word essay about the history of bridges. Variant {i}."}]})
            n = (d.get("usage") or {}).get("completion_tokens") or 0
            out["cost_usd"] += (d.get("usage") or {}).get("cost") or 0
            if n and s:
                rates.append(n / s)
        out["decode_tps"] = round(statistics.median(rates), 1) if rates else None
    except Exception as e:  # noqa: BLE001
        out["decode_tps"] = None
        out["errors"].append(f"decode: {str(e)[:160]}")
    # 4. cache reporting: same long prompt twice; does usage carry cached_tokens on the repeat?
    try:
        body = {"model": model, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": LONG_PROMPT + " (cache probe)"}]}
        d0, _ = chat(base, key, body)
        d, _ = chat(base, key, body)
        u = d.get("usage") or {}
        out["cost_usd"] += ((d0.get("usage") or {}).get("cost") or 0) + (u.get("cost") or 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = u.get("cache_read_input_tokens")
        out["cache_reported"] = cached is not None
        out["cache_probe_cached_tokens"] = cached
        out["cache_probe_prompt_tokens"] = u.get("prompt_tokens")
    except Exception as e:  # noqa: BLE001
        out["cache_reported"] = None
        out["errors"].append(f"cache: {str(e)[:160]}")
    out["wall_s"] = round(time.time() - t_leg, 1)
    out["cost_usd"] = round(out["cost_usd"], 4) if out["cost_usd"] else None
    return out


# ---------------------------------------------------------------- subprocess legs
def redact(s: str) -> str:
    """Endpoint hosts are tailnet addresses; logs under results/ are committed."""
    return re.sub(r"(https?://)[^/\s]+", r"\1<host>", s)


def run(cmd: list[str], env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    print("  $", redact(" ".join(str(c) for c in cmd))[:220], flush=True)
    # Children decode the agents' UTF-8 output; without this a Windows console
    # codec turns one curly quote into a failed leg.
    env = {**(env or os.environ), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        # A leg that hangs must cost the leg, not the gate: the other three
        # legs' files are already on disk and the verdict should say so.
        tail = redact(((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))[-800:])
        print(f"  leg timed out after {timeout}s", flush=True)
        return 124, f"timeout after {timeout}s\n{tail}"
    tail = redact((p.stdout + p.stderr)[-1500:])
    if p.returncode != 0:
        print(tail, flush=True)
    return p.returncode, tail


def leg_coding(a, out_dir: Path, tag: str) -> dict:
    out = out_dir / f"{tag}-coding.json"
    rc, tail = run([sys.executable, HERE / "polyglot_bench.py", "--bench", a.bench, "--lang", "python",
                    "--exercises", *a.exercises, "--agents", "jcode", "--jcode", a.jcode,
                    "--jcode-profile", a.jcode_profile, "--model", a.model, "--timeout", a.timeout,
                    "--label", "jcode", "--out", out])
    if not out.exists():
        return {"error": f"polyglot_bench exit {rc}: {tail[-300:]}", "solved": 0, "n": len(a.exercises)}
    s = json.load(open(out, encoding="utf-8"))["summary"]["jcode"]
    return {"solved": s["solved"], "n": s["n"], "median_wall_s": s["median_wall_s"], "total_wall_s": s["total_wall_s"],
            "input_tokens": s["input_tokens"], "output_tokens": s.get("output_tokens"), "cached_tokens": s["cached_tokens"],
            "cost_usd": cost_of(a.pricing, s["input_tokens"], s.get("output_tokens"), s["cached_tokens"]),
            "errors": s["errors"], "file": out.name}


def leg_agentic(a, out_dir: Path, tag: str) -> dict:
    res: dict = {}
    out = out_dir / f"{tag}-docbench.json"
    rc, tail = run([sys.executable, HERE / "docbench.py", "--tasks", *a.doc_tasks, "--model", a.model,
                    "--jcode", a.jcode, "--jcode-profile", a.jcode_profile, "--timeout", a.timeout,
                    "--label", "jcode", "--out", out])
    if out.exists():
        s = json.load(open(out, encoding="utf-8"))["summary"]
        res["doc"] = {"solved": s["solved"], "n": s["n"], "checks_passed": s["checks_passed"],
                      "checks_total": s["checks_total"], "total_wall_s": s["total_wall_s"], "errors": s["errors"],
                      "input_tokens": s.get("input_tokens"), "output_tokens": s.get("output_tokens"), "cached_tokens": s.get("cached_tokens"),
                      "cost_usd": cost_of(a.pricing, s.get("input_tokens"), s.get("output_tokens"), s.get("cached_tokens")),
                      "file": out.name}
    else:
        res["doc"] = {"error": f"docbench exit {rc}: {tail[-300:]}", "solved": 0, "n": len(a.doc_tasks)}
    out2 = out_dir / f"{tag}-stateful.json"
    rc, tail = run([sys.executable, HERE / "ab_compare.py", "--jcode", a.jcode, "--baseline-provider", a.jcode_profile,
                    "--baseline-model", a.model, "--tasks", *a.ab_tasks, "--baseline-only", "--timeout", a.timeout,
                    "--out", out2])
    if out2.exists():
        rows = json.load(open(out2, encoding="utf-8"))
        ti, to, tc = (sum(r.get(k) or 0 for r in rows) for k in ("input_tokens", "output_tokens", "cached_tokens"))
        res["stateful"] = {"solved": sum(1 for r in rows if r["solved"]),
                           "followup_solved": sum(1 for r in rows if r.get("followup_solved")),
                           "n": len(rows), "wall_s": round(sum(r["wall_s"] for r in rows), 1),
                           "input_tokens": ti, "output_tokens": to, "cached_tokens": tc,
                           "cost_usd": cost_of(a.pricing, ti, to, tc),
                           "errors": sum(1 for r in rows if r["error"]), "file": out2.name}
    else:
        res["stateful"] = {"error": f"ab_compare exit {rc}: {tail[-300:]}", "solved": 0, "followup_solved": 0, "n": len(a.ab_tasks)}
    return res


def leg_memory(a, base: str, key: str | None, out_dir: Path, tag: str) -> dict:
    out = out_dir / f"{tag}-memory.json"
    env = dict(os.environ)
    if key:
        env["LH_API_KEY"] = key
    cmd = [sys.executable, HERE / "long_horizon.py", "--base", base, "--model", a.model, "--arm", "direct-full",
           # 2048, not 120: on the gateway, reasoning tokens count against
           # max_tokens, and a 120 cap returned "HR-229" with finish_reason
           # length. The fleet's own rule for reasoning models is >= 2048.
           "--turns", a.lh_turns, "--facts", a.lh_facts, "--max-tokens", "2048", "--timeout", "180", "--out", out]
    if a.reasoning_effort:
        cmd += ["--reasoning-effort", a.reasoning_effort]
    if EXTRA_BODY.get("usage"):
        cmd += ["--usage-include"]
    rc, tail = run(cmd, env=env)
    if not out.exists():
        return {"error": f"long_horizon exit {rc}: {tail[-300:]}", "recall": 0, "probes": a.lh_facts}
    s = json.load(open(out, encoding="utf-8"))["summary"]
    return {"recall": s["recall"], "probes": s["probes"], "mean_latency_s": s["mean_latency_s"],
            "p90_latency_s": s["p90_latency_s"], "prompt_tokens": s["prompt_tokens"], "cached_tokens": s["cached_tokens"],
            "completion_tokens": s.get("completion_tokens"),
            "cost_usd": s.get("cost_usd") if s.get("cost_usd") is not None
            else cost_of(a.pricing, s["prompt_tokens"], s.get("completion_tokens"), s["cached_tokens"]),
            "truncated": s.get("truncated"), "errors": s["errors"], "total_s": s["total_s"], "file": out.name}


# ---------------------------------------------------------------- verdict
def fold(model: str, legs: dict) -> dict:
    e, c, ag, m = legs.get("engine", {}), legs.get("coding", {}), legs.get("agentic", {}), legs.get("memory", {})
    doc, st = ag.get("doc", {}), ag.get("stateful", {})
    got = (c.get("solved", 0) + doc.get("solved", 0) + st.get("solved", 0) + st.get("followup_solved", 0) + m.get("recall", 0))
    tot = (c.get("n", 0) + doc.get("n", 0) + 2 * st.get("n", 0) + m.get("probes", 0))
    costs = [x.get("cost_usd") for x in (e, c, doc, st, m) if x.get("cost_usd") is not None]
    n_tasks = c.get("n", 0) + doc.get("n", 0) + st.get("n", 0)
    task_in = sum(x.get("input_tokens") or 0 for x in (c, doc, st))
    task_out = sum(x.get("output_tokens") or 0 for x in (c, doc, st))
    task_cost = sum(x.get("cost_usd") or 0 for x in (c, doc, st) if x.get("cost_usd") is not None)
    return {
        "model": model,
        "reasoning_effort": legs.get("_effort"),
        "gate_cost_usd": round(sum(costs), 4) if costs else None,
        "cost_per_task_usd": round(task_cost / n_tasks, 4) if (n_tasks and any(x.get("cost_usd") is not None for x in (c, doc, st))) else None,
        "tokens_per_task": round((task_in + task_out) / n_tasks) if n_tasks and task_in else None,
        "output_tokens_per_task": round(task_out / n_tasks) if n_tasks and task_out else None,
        "correctness": f"{got}/{tot}",
        "correctness_rate": round(got / tot, 3) if tot else None,
        "coding": f"{c.get('solved', 0)}/{c.get('n', 0)}",
        "agentic": f"{doc.get('solved', 0) + st.get('solved', 0) + st.get('followup_solved', 0)}/{doc.get('n', 0) + 2 * st.get('n', 0)}",
        "memory": f"{m.get('recall', 0)}/{m.get('probes', 0)}",
        "tool_call": e.get("tool_call"),
        "coding_median_wall_s": c.get("median_wall_s"),
        "decode_tps": e.get("decode_tps"),
        "ttft_s": e.get("ttft_s"),
        "cache_reported": e.get("cache_reported"),
        "errors": sum(x.get("errors", 0) if isinstance(x.get("errors"), int) else len(x.get("errors", [])) for x in (e, c, doc, st, m)),
    }


def verdict_line(v: dict) -> str:
    tc = "✓" if v["tool_call"] else "✗"
    cache = {True: "yes", False: "no", None: "?"}[v["cache_reported"]]
    eff = f" ({v['reasoning_effort']})" if v.get("reasoning_effort") else ""
    cost = f"  | ${v['cost_per_task_usd']}/task · {v['tokens_per_task']} tok/task · gate ${v['gate_cost_usd']}" if v.get("cost_per_task_usd") is not None else \
           (f"  | {v['tokens_per_task']} tok/task" if v.get("tokens_per_task") else "")
    return (f"{v['model'] + eff:<34} correctness {v['correctness']:>6}  coding {v['coding']}  agentic {v['agentic']}  memory {v['memory']}  "
            f"| speed {v['coding_median_wall_s'] or '?'} s/task · {v['decode_tps'] or '?'} tok/s · TTFT {v['ttft_s'] or '?'} s  "
            f"| tools {tc}  cache {cache}  errors {v['errors']}{cost}")


def report(dir_: str) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(dir_, "gate-*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        rows.append((d["verdict"], d.get("date"), os.path.basename(f)))
    rows.sort(key=lambda r: (-(r[0]["correctness_rate"] or 0), r[0]["coding_median_wall_s"] or 1e9))
    print("| model | effort | correctness | coding | agentic | memory | s/task | tok/task | $/task | gate $ | tok/s | TTFT | tools | cache | errors | date |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v, date, _ in rows:
        print(f"| {v['model']} | {v.get('reasoning_effort') or ''} | {v['correctness']} | {v['coding']} | {v['agentic']} | {v['memory']} | {v['coding_median_wall_s']} | "
              f"{v.get('tokens_per_task') or ''} | {v.get('cost_per_task_usd') if v.get('cost_per_task_usd') is not None else ''} | "
              f"{v.get('gate_cost_usd') if v.get('gate_cost_usd') is not None else ''} | "
              f"{v['decode_tps']} | {v['ttft_s']} | {'✓' if v['tool_call'] else '✗'} | "
              f"{ {True: 'yes', False: 'no', None: '?'}[v['cache_reported']] } | {v['errors']} | {date} |")


def refold(path: str) -> None:
    """Recompute a gate's verdict from its leg files, after one leg was re-run
    on its own (e.g. long_horizon.py with a corrected max_tokens)."""
    p = Path(path)
    d = json.load(open(p, encoding="utf-8"))
    legs = d["legs"]
    if "memory" in legs and legs["memory"].get("file"):
        s = json.load(open(p.parent / legs["memory"]["file"], encoding="utf-8"))["summary"]
        legs["memory"].update({"recall": s["recall"], "probes": s["probes"], "mean_latency_s": s["mean_latency_s"],
                               "p90_latency_s": s["p90_latency_s"], "prompt_tokens": s["prompt_tokens"],
                               "cached_tokens": s["cached_tokens"], "errors": s["errors"], "total_s": s["total_s"],
                               "truncated": s.get("truncated")})
    ag = legs.get("agentic", {})
    st_file = p.parent / f"{p.stem[5:]}-stateful.json"
    if "agentic" in legs and st_file.exists():
        rows = json.load(open(st_file, encoding="utf-8"))
        ag["stateful"] = {"solved": sum(1 for r in rows if r["solved"]),
                          "followup_solved": sum(1 for r in rows if r.get("followup_solved")),
                          "n": len(rows), "wall_s": round(sum(r["wall_s"] for r in rows), 1),
                          "errors": sum(1 for r in rows if r["error"]), "file": st_file.name}
    doc_file = p.parent / f"{p.stem[5:]}-docbench.json"
    if "agentic" in legs and doc_file.exists():
        s = json.load(open(doc_file, encoding="utf-8"))["summary"]
        ag["doc"] = {"solved": s["solved"], "n": s["n"], "checks_passed": s["checks_passed"],
                     "checks_total": s["checks_total"], "total_wall_s": s["total_wall_s"], "errors": s["errors"], "file": doc_file.name}
    if "coding" in legs and legs["coding"].get("file"):
        s = json.load(open(p.parent / legs["coding"]["file"], encoding="utf-8"))["summary"]["jcode"]
        legs["coding"].update({"solved": s["solved"], "n": s["n"], "median_wall_s": s["median_wall_s"],
                               "total_wall_s": s["total_wall_s"], "input_tokens": s["input_tokens"],
                               "cached_tokens": s["cached_tokens"], "errors": s["errors"]})
    d["verdict"] = fold(d["model"], legs)
    d.setdefault("refolded", []).append(dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(verdict_line(d["verdict"]))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    # The verdict line carries ✓/✗; a cp1252 console would otherwise abort the
    # whole run at the last print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="served model name")
    ap.add_argument("--jcode-profile", default="qgillm", help="[providers.X] in ~/.jcode/config.toml; base_url and key come from it")
    ap.add_argument("--base", default=None, help="override the endpoint base URL (.../v1) for the direct-API legs")
    ap.add_argument("--api-key", default=None, help="override the bearer for the direct-API legs")
    ap.add_argument("--bench", default=None, help="Aider polyglot-benchmark checkout (coding leg)")
    ap.add_argument("--jcode", default=str(DEFAULT_JCODE))
    ap.add_argument("--exercises", nargs="*", default=CODING_EXERCISES)
    ap.add_argument("--doc-tasks", nargs="*", default=DOC_TASKS)
    ap.add_argument("--ab-tasks", nargs="*", default=AB_TASKS)
    ap.add_argument("--lh-turns", type=int, default=LH_TURNS)
    ap.add_argument("--lh-facts", type=int, default=LH_FACTS)
    ap.add_argument("--legs", nargs="*", default=["engine", "coding", "agentic", "memory"],
                    choices=["engine", "coding", "agentic", "memory"])
    ap.add_argument("--timeout", type=int, default=600, help="seconds per agent task")
    ap.add_argument("--reasoning-effort", default=None,
                    help="sent as reasoning_effort on the direct-API legs; the jcode legs take it from the profile's extra_body")
    ap.add_argument("--pricing", default="auto", choices=["auto", "openrouter", "none"],
                    help="cost per leg from per-token rates: auto = OpenRouter when the endpoint is openrouter.ai")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "gate"))
    ap.add_argument("--tag", default=None, help="file prefix (default: slug of the model)")
    ap.add_argument("--report", default=None, help="print a table over every gate-*.json in this directory and exit")
    ap.add_argument("--refold", default=None, help="recompute the verdict of this gate-*.json from its leg files and exit")
    a = ap.parse_args()

    if a.report:
        report(a.report)
        return
    if a.refold:
        refold(a.refold)
        return
    if not a.model:
        ap.error("--model is required")
    if "coding" in a.legs and not a.bench:
        ap.error("--bench is required for the coding leg")

    base, key = jcode_provider(a.jcode_profile)
    base = a.base or base
    if not base:
        ap.error(f"no base_url for profile {a.jcode_profile!r}; pass --base")
    is_openrouter = "openrouter.ai" in base

    # Key resolution, in order: --api-key; the profile's own api_key_env from
    # the shell or the repo .env (never printed); then the endpoint-specific
    # fallback. The LiteLLM master key must never reach OpenRouter -- it did
    # once, as a 401 on every direct leg.
    key_env = jcode_key_env(a.jcode_profile)
    if key_env and not os.environ.get(key_env) and repo_env(key_env):
        # jcode reads it from its environment too.
        os.environ[key_env] = repo_env(key_env)
    key = a.api_key or (os.environ.get(key_env) if key_env else None) or key
    if not key:
        key = (os.environ.get("OPENROUTER_API_KEY") or repo_env("OPENROUTER_API_KEY")) if is_openrouter \
            else (os.environ.get("LITELLM_MASTER_KEY") or repo_env("LITELLM_MASTER_KEY"))
    if a.reasoning_effort:
        EXTRA_BODY["reasoning_effort"] = a.reasoning_effort
    if is_openrouter:
        EXTRA_BODY["usage"] = {"include": True}
    a.pricing = openrouter_pricing(a.model, key) if (a.pricing == "openrouter" or (a.pricing == "auto" and is_openrouter)) else None
    if a.pricing:
        print(f"   pricing: ${a.pricing['prompt'] * 1e6:.2f}/M in · ${a.pricing['completion'] * 1e6:.2f}/M out · "
              f"${a.pricing['cache_read'] * 1e6:.3f}/M cache read", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or slug(a.model)
    started = time.time()
    spend_before = openrouter_key_usage(key) if is_openrouter else None
    print(f"== gate: {a.model} via {a.jcode_profile} ({re.sub(r'//[^/]+', '//<host>', base)})  legs: {', '.join(a.legs)}", flush=True)

    legs: dict = {}
    if "engine" in a.legs:
        print("-- engine", flush=True)
        legs["engine"] = leg_engine(base, key, a.model)
        print("   ", {k: v for k, v in legs["engine"].items() if k != "base"}, flush=True)
        if not legs["engine"]["tool_call"] and "coding" in a.legs:
            print("   tool call did not come back structured; the agent legs will run anyway but expect failures", flush=True)
    if "coding" in a.legs:
        print("-- coding", flush=True)
        legs["coding"] = leg_coding(a, out_dir, tag)
        print("   ", legs["coding"], flush=True)
    if "agentic" in a.legs:
        print("-- agentic", flush=True)
        legs["agentic"] = leg_agentic(a, out_dir, tag)
        print("   ", legs["agentic"], flush=True)
    if "memory" in a.legs:
        print("-- memory", flush=True)
        legs["memory"] = leg_memory(a, base, key, out_dir, tag)
        print("   ", legs["memory"], flush=True)

    legs["_effort"] = a.reasoning_effort
    if spend_before is not None:
        spend_after = openrouter_key_usage(key)
        if spend_after is not None:
            # Exact, reasoning included. Only meaningful when nothing else used
            # the key during the run; parallel gates share it, so the priced
            # per-leg figures stay the per-model number and this is the check.
            legs["_key_spend_delta_usd"] = round(spend_after - spend_before, 4)
    v = fold(a.model, legs)
    v["key_spend_delta_usd"] = legs.get("_key_spend_delta_usd")
    doc = {"model": a.model, "profile": a.jcode_profile, "endpoint": re.sub(r"//[^/]+", "//<host>", base),
           "reasoning_effort": a.reasoning_effort, "pricing": a.pricing,
           "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "gate_wall_s": round(time.time() - started, 1),
           "config": {"exercises": a.exercises, "doc_tasks": a.doc_tasks, "ab_tasks": a.ab_tasks,
                      "lh_turns": a.lh_turns, "lh_facts": a.lh_facts, "timeout": a.timeout},
           "legs": legs, "verdict": v}
    out = out_dir / f"gate-{tag}.json"
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print()
    print(verdict_line(v))
    print(f"gate wall {doc['gate_wall_s']} s -> {out}")


if __name__ == "__main__":
    main()
