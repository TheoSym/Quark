#!/usr/bin/env python3
"""A/B the QGI-2 harness against baseline jcode, end to end, on coding tasks.

    python bench/ab_compare.py \
      --baseline-provider openrouter --baseline-model qwen/qwen3-9b \
      --qgi2-model qgi2/builder-traceable \
      --repeats 3

The only thing that differs between the two arms is which provider jcode talks
to. Same jcode binary, same tools, same tasks, same verification.

# The one rule that makes this mean anything

**Point both arms at the same underlying model.** Baseline goes to it directly
(through LiteLLM, say); QGI-2 goes to it through the harness. Otherwise you are
measuring the model, not the harness, and a QGI-2 arm on a 9B against a baseline
on Claude tells you only what you already knew.

# What is measured

Task success is the outcome; everything else is diagnosis. A harness that halves
tokens while failing more tasks has not helped.

Cache hit rate and token counts come from jcode's own `--json` output, so they
are the same numbers jcode itself reports — not something this script computes.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import by_name  # noqa: E402


class Arm:
    """One side of the comparison."""

    def __init__(self, label, provider, model, extra_env=None):
        self.label = label
        self.provider = provider
        self.model = model
        self.extra_env = extra_env or {}

    def jcode_args(self):
        args = []
        if self.provider:
            args += ["--provider", self.provider]
        if self.model:
            args += ["--model", self.model]
        return args


def run_task(jcode, arm, task, workdir, timeout):
    """Run one task in one arm. Returns a result dict."""
    for name, body in task.files.items():
        p = workdir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    env = {**os.environ, **arm.extra_env}
    result = {
        "arm": arm.label,
        "task": task.name,
        "solved": False,
        "followup_solved": None,
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "wall_s": 0.0,
        "error": None,
    }

    def one_turn(prompt):
        cmd = [jcode, "run", "--json", *arm.jcode_args(), prompt]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result["error"] = f"timeout after {timeout}s"
            return None
        finally:
            result["wall_s"] += time.time() - t0

        result["turns"] += 1
        if proc.returncode != 0:
            # Keep the tail: the useful part of a jcode failure is at the end.
            result["error"] = (proc.stderr or proc.stdout or "")[-400:].strip()
            return None
        # jcode --json emits NDJSON: one event per line, with a final
        # {"type": "done", ...} line carrying the usage. Parsing the whole
        # stdout as one document fails on the first newline.
        payload = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "done":
                payload = event
        if payload is None:
            result["error"] = "jcode --json emitted no 'done' event"
            return None

        usage = payload.get("usage") or {}
        result["input_tokens"] += usage.get("input_tokens") or 0
        result["output_tokens"] += usage.get("output_tokens") or 0
        result["cached_tokens"] += usage.get("cache_read_input_tokens") or 0
        return payload

    if one_turn(task.prompt) is None:
        return result
    result["solved"] = verify(task.verify, workdir)

    if task.followup:
        if one_turn(task.followup) is None:
            return result
        result["followup_solved"] = verify(task.followup_verify, workdir)

    return result


def verify(command, workdir):
    """Objective check. Exit 0 means solved."""
    if not command:
        return None
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def summarize(results, arm_label):
    rows = [r for r in results if r["arm"] == arm_label]
    if not rows:
        return None
    solved = sum(1 for r in rows if r["solved"])
    followups = [r for r in rows if r["followup_solved"] is not None]
    prompt_total = sum(r["input_tokens"] for r in rows)
    cached_total = sum(r["cached_tokens"] for r in rows)
    return {
        "arm": arm_label,
        "runs": len(rows),
        "solved": solved,
        "solve_rate": solved / len(rows),
        "followup_rate": (
            sum(1 for r in followups if r["followup_solved"]) / len(followups)
            if followups
            else None
        ),
        "median_turns": statistics.median(r["turns"] for r in rows),
        "total_tokens": prompt_total + sum(r["output_tokens"] for r in rows),
        # Guard the division: a run where every request failed before reaching
        # the model has no prompt tokens, and reporting 0% would look like a
        # cache problem rather than a connection one.
        "cache_hit_rate": (cached_total / prompt_total) if prompt_total else None,
        "median_wall_s": statistics.median(r["wall_s"] for r in rows),
        "errors": sum(1 for r in rows if r["error"]),
    }


def fmt(v, spec="{:.2f}"):
    return "n/a" if v is None else spec.format(v)


def print_table(summaries):
    print()
    header = f"{'arm':<22} {'solved':>10} {'2nd turn':>9} {'turns':>6} {'tokens':>10} {'cache':>7} {'wall s':>8} {'err':>4}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        if not s:
            continue
        print(
            f"{s['arm']:<22} "
            f"{s['solved']:>3}/{s['runs']:<3} {s['solve_rate'] * 100:>5.0f}% "
            f"{fmt(s['followup_rate'] and s['followup_rate'] * 100, '{:.0f}%'):>9} "
            f"{s['median_turns']:>6.0f} "
            f"{s['total_tokens']:>10,} "
            f"{fmt(s['cache_hit_rate'] and s['cache_hit_rate'] * 100, '{:.0f}%'):>7} "
            f"{s['median_wall_s']:>8.1f} "
            f"{s['errors']:>4}"
        )
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jcode", default="jcode", help="path to the jcode binary")
    ap.add_argument("--baseline-provider", required=True)
    ap.add_argument("--baseline-model", default=None)
    ap.add_argument("--qgi2-provider", default="qgi2")
    ap.add_argument("--qgi2-model", default="qgi2/builder-traceable")
    ap.add_argument("--repeats", type=int, default=1, help="runs per task per arm")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per turn")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the scratch repos")
    ap.add_argument("--out", default=None, help="write raw results as JSON")
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="record a baseline without QGI-2 running",
    )
    args = ap.parse_args()

    if not shutil.which(args.jcode) and not Path(args.jcode).exists():
        raise SystemExit(f"jcode not found: {args.jcode}")

    arms = [Arm("baseline", args.baseline_provider, args.baseline_model)]
    if not args.baseline_only:
        arms.append(Arm("qgi2", args.qgi2_provider, args.qgi2_model))

    tasks = by_name(args.tasks)
    root = Path(tempfile.mkdtemp(prefix="qgi2-ab-"))
    print(f"scratch: {root}")
    print(f"tasks:   {', '.join(t.name for t in tasks)}")
    print(f"arms:    {', '.join(a.label for a in arms)}  x{args.repeats}")

    results = []
    for rep in range(args.repeats):
        for task in tasks:
            for arm in arms:
                # A fresh copy per run: a task solved in run 1 must not be
                # already-solved in run 2.
                wd = root / f"{arm.label}-{task.name}-{rep}"
                wd.mkdir(parents=True)
                print(f"  [{rep + 1}/{args.repeats}] {arm.label:<9} {task.name} ...", end="", flush=True)
                r = run_task(args.jcode, arm, task, wd, args.timeout)
                results.append(r)
                mark = "PASS" if r["solved"] else ("ERR " if r["error"] else "fail")
                print(f" {mark}  {r['wall_s']:.0f}s")
                if r["error"]:
                    print(f"        {r['error'].splitlines()[0][:120]}")

    summaries = [summarize(results, a.label) for a in arms]
    print_table(summaries)

    if len(summaries) == 2 and all(summaries):
        b, q = summaries
        print("delta (qgi2 vs baseline):")
        print(f"  solve rate   {(q['solve_rate'] - b['solve_rate']) * 100:+.0f} points")
        if b["total_tokens"]:
            print(f"  tokens       {(q['total_tokens'] / b['total_tokens'] - 1) * 100:+.0f}%")
        if b["median_wall_s"]:
            print(f"  wall clock   {(q['median_wall_s'] / b['median_wall_s'] - 1) * 100:+.0f}%")
        print()
        print("Read the solve rate first. A harness that halves tokens while")
        print("failing more tasks has not helped.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"raw results -> {args.out}")

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"scratch kept at {root}")


if __name__ == "__main__":
    main()
