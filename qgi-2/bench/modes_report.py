#!/usr/bin/env python
"""Fold a coding-modes matrix directory into one table: model x arm, with
solved, median and total wall, tokens, and per-exercise walls.

    python modes_report.py results/modes-2026-09-14 [--md|--json]
"""
from __future__ import annotations

import glob
import json
import os
import sys

ARMS = ["baseline", "safe-efficient", "exa-reuse", "exa-reuse+tool", "mem", "mem+exa-tool"]
ALIASES = {"exa-reuse-tool": "exa-reuse+tool"}  # the 13 Sep file names


def load(dir_: str) -> dict:
    cells: dict = {}
    for f in sorted(glob.glob(os.path.join(dir_, "*.json"))):
        base = os.path.basename(f)[:-5]
        try:
            key, arm = base.split("-", 1)
        except ValueError:
            continue
        arm = ALIASES.get(arm, arm)
        if arm not in ARMS:
            # "ds41-exa-reuse-tool" splits as key="ds41", arm="exa-reuse-tool" -> alias above;
            # "fable-low-baseline" splits as key="fable", arm="low-baseline" -> re-split.
            parts = base.split("-")
            for i in range(1, len(parts)):
                k2, a2 = "-".join(parts[:i]), "-".join(parts[i:])
                a2 = ALIASES.get(a2, a2)
                if a2 in ARMS:
                    key, arm = k2, a2
                    break
            else:
                continue
        d = json.load(open(f, encoding="utf-8"))
        s = list(d["summary"].values())[0] if d.get("summary") else {}
        cells[(key, arm)] = {
            "model": d.get("model"), "status": s.get("status"), "solved": s.get("solved"), "n": s.get("n"),
            "median": s.get("median_wall_s"), "total": s.get("total_wall_s"), "in": s.get("input_tokens"),
            "out": s.get("output_tokens"), "cached": s.get("cached_tokens"), "errors": s.get("errors"),
            "per": {r["exercise"]: (r["wall_s"], r["solved"]) for r in d.get("results", [])},
            "note": d.get("note") or d.get("copied_from"),
        }
    return cells


def main() -> None:
    dir_ = sys.argv[1]
    cells = load(dir_)
    keys = []
    for k, _ in cells:
        if k not in keys:
            keys.append(k)
    if "--json" in sys.argv:
        print(json.dumps({f"{k}/{a}": v for (k, a), v in cells.items()}, indent=1))
        return
    print("| model | arm | solved | median s | total s | input tok | output tok | cache | bowling | forth | wordy | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for a in ARMS:
            c = cells.get((k, a))
            if not c:
                print(f"| {k} | {a} | - | - | - | - | - | - | - | - | - | not run |")
                continue
            if c["status"] == "BLOCKED":
                print(f"| {c['model']} | {a} | BLOCKED | - | - | - | - | - | - | - | - | {c['note']} |")
                continue
            cache = f"{100 * (c['cached'] or 0) / c['in']:.0f}%" if c.get("in") else "-"
            per = " | ".join(
                (f"{c['per'][e][0]}{'' if c['per'][e][1] else ' ✗'}" if e in c["per"] else "-")
                for e in ("bowling", "forth", "wordy"))
            print(f"| {c['model']} | {a} | {c['solved']}/{c['n']} | {c['median']} | {c['total']} | {c['in']} | {c['out']} | {cache} | {per} | {(c['note'] or '')[:40]} |")


if __name__ == "__main__":
    main()
