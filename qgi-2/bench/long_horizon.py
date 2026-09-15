#!/usr/bin/env python
"""Non-coding long-horizon conversation test: 100 turns, facts planted early,
recall probed late, through a sliding-window chat client.

Why a window: a real chat client compacts or truncates history. With the full
transcript in context, memory has nothing to add; with a window, facts that
scrolled out are only recoverable through the QGI-2 memory proxy. Three arms:

  direct-window   engine directly, last WINDOW messages only  (memory: none)
  proxy-window    same window through `qgi2 serve --proxy --compact`
  direct-full     engine directly, full history               (upper bound, cost of growth)

Scoring: each recall probe asks for one planted value; a hit is the value
appearing in the reply. Latency is wall time per turn, plus prompt/cached
tokens from the usage block.

Example:
  python long_horizon.py --base http://127.0.0.1:18035/v1 --model Qwen3.8-Flash-Next-NVFP4 \
      --arm direct-window --out lh-qwen38-direct-window.json
  python long_horizon.py --base http://127.0.0.1:8792/v1 --model Qwen3.8-Flash-Next-NVFP4 \
      --arm proxy-window --session lh-qwen38-1 --out lh-qwen38-proxy-window.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request

SYSTEM = ("You are a concise project assistant for a small robotics startup called Halden Robotics. "
          "Answer in one or two sentences. When asked for a fact you were told earlier, state it exactly.")

# 20 planted facts: (key phrase for the probe, value, planting sentence)
FACTS = [
    ("the internal codename of the warehouse robot", "Kestrel", "For reference, the internal codename of the warehouse robot is Kestrel."),
    ("the name of our lead firmware engineer", "Priya Vantarakis", "Our lead firmware engineer is Priya Vantarakis; she joined in March."),
    ("the torque rating of the gripper motor", "4.2 Nm", "The gripper motor is rated at 4.2 Nm continuous torque."),
    ("the ID of the support ticket about the lidar dropout", "HR-2291", "The lidar dropout issue is tracked as support ticket HR-2291."),
    ("the city where our second assembly line is", "Tallinn", "Our second assembly line is in Tallinn, opened last quarter."),
    ("the battery cell supplier's name", "Norvolt Cells", "We source battery cells from Norvolt Cells under a two-year contract."),
    ("the maximum payload of the Kestrel", "38 kg", "The Kestrel's maximum payload is 38 kg."),
    ("the date of the next safety audit", "14 November", "The next safety audit is scheduled for 14 November."),
    ("the name of the customer pilot in Rotterdam", "Veldhaven Logistics", "The Rotterdam pilot customer is Veldhaven Logistics."),
    ("the firmware version currently in the field", "3.7.2", "Fleet units are running firmware 3.7.2 in the field."),
    ("the wheel diameter of the base platform", "210 mm", "The base platform uses 210 mm wheels."),
    ("the CAN bus bitrate we standardised on", "500 kbit/s", "We standardised the CAN bus at 500 kbit/s across all boards."),
    ("the name of the finance lead", "Tomasz Gierek", "Tomasz Gierek is our finance lead and signs off purchase orders."),
    ("the color code of the emergency stop housing", "RAL 3020", "The emergency stop housing is painted RAL 3020."),
    ("the Wi-Fi SSID used on the test floor", "HALDEN-FLOOR-B", "The test floor Wi-Fi SSID is HALDEN-FLOOR-B."),
    ("the depth camera model on the mast", "Orbis D455X", "The mast carries an Orbis D455X depth camera."),
    ("the number of units in the first production batch", "120 units", "The first production batch is 120 units."),
    ("the name of the simulation environment we use", "Gazebo Harmonic", "We simulate in Gazebo Harmonic before hardware trials."),
    ("the monthly cloud budget cap", "8,500 euros", "The monthly cloud budget cap is 8,500 euros."),
    ("the acronym of the safety standard we certify against", "ISO 3691-4", "We certify against ISO 3691-4 for driverless trucks."),
]

FILLER = [
    "Give me one sentence of encouragement for the team this week.",
    "What is a reasonable default for a heartbeat interval on a mobile robot's watchdog?",
    "Suggest a short name for a Friday demo session.",
    "In one sentence, why do we log timestamps in UTC?",
    "What is a good rule of thumb for the number of test cycles before a firmware release?",
    "Suggest one agenda item for tomorrow's standup.",
    "In one sentence, what is the point of a dead-man switch on a robot?",
    "Give me a one-line summary of what a Kalman filter does.",
    "What is one risk of running lidar in direct sunlight?",
    "Suggest a polite one-line reply to a supplier who is late.",
    "In one sentence, what does 'fail-safe' mean for a brake?",
    "Name one metric to watch on a warehouse robot fleet dashboard.",
]


def build_script(seed: int = 7, turns_total: int = 100, n_facts: int = len(FACTS)) -> list[dict]:
    """`turns_total` turns: plants in the first 55 %, probes in the last 55 %, filler elsewhere.

    The default (100 turns, 20 facts) is the full measurement; the gate runs a
    shorter script with the same shape (e.g. 40 turns, 8 facts), which is still
    long enough for a 12-message window to have dropped every plant.
    """
    rng = random.Random(seed)
    facts = FACTS[:n_facts]
    turns: list[dict] = []
    plant_hi = max(n_facts + 1, int(turns_total * 0.55) + 1)
    probe_lo = max(1, int(turns_total * 0.45) + 1)
    plant_turns = sorted(rng.sample(range(1, plant_hi), len(facts)))
    probe_turns = sorted(rng.sample(range(probe_lo, turns_total + 1), len(facts)))
    plants = {t: f for t, f in zip(plant_turns, facts)}
    order = list(range(len(facts)))
    rng.shuffle(order)
    probes = {t: facts[i] for t, i in zip(probe_turns, order)}
    for t in range(1, turns_total + 1):
        if t in plants and t not in probes:
            key, val, sentence = plants[t]
            turns.append({"turn": t, "kind": "plant", "key": key, "value": val,
                          "text": f"{sentence} Just acknowledge in a few words."})
        elif t in probes:
            key, val, _ = probes[t]
            turns.append({"turn": t, "kind": "probe", "key": key, "value": val,
                          "text": f"Quick check: what is {key}? Answer with the exact value."})
        else:
            turns.append({"turn": t, "kind": "filler", "text": rng.choice(FILLER)})
    # A plant that collided with a probe turn was dropped; make sure every probed
    # fact was planted earlier, otherwise drop the probe.
    planted = {x["value"] for x in turns if x["kind"] == "plant"}
    for x in turns:
        if x["kind"] == "probe" and x["value"] not in planted:
            x["kind"] = "filler"; x["text"] = FILLER[0]
    return turns


def chat(base: str, model: str, messages: list[dict], max_tokens: int, timeout: int, extra: dict) -> tuple[str, dict, float]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    extra = dict(extra)
    auth = extra.pop("_auth", None)
    body.update(extra)
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    choice = d["choices"][0]
    usage = dict(d.get("usage") or {})
    usage["finish_reason"] = choice.get("finish_reason")
    return (choice["message"].get("content") or ""), usage, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["direct-window", "proxy-window", "direct-full"])
    ap.add_argument("--session", default=None, help="proxy arm: session key appended as model@session")
    ap.add_argument("--window", type=int, default=12, help="messages kept (user+assistant) in window arms")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--turns", type=int, default=100, help="length of the script (default: the full 100)")
    ap.add_argument("--facts", type=int, default=len(FACTS), help="facts planted and probed (default: all 20)")
    ap.add_argument("--api-key", default=None, help="bearer for the endpoint (or env LH_API_KEY); local engines need none")
    ap.add_argument("--reasoning-effort", default=None, help="sent as reasoning_effort on every request")
    ap.add_argument("--usage-include", action="store_true", help='send usage: {include: true} (OpenRouter returns usage.cost)')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = f"{args.model}@{args.session}" if (args.arm == "proxy-window" and args.session) else args.model
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    if args.usage_include:
        extra["usage"] = {"include": True}
    key = args.api_key or os.environ.get("LH_API_KEY")
    if key:
        extra["_auth"] = key
    script = build_script(turns_total=args.turns, n_facts=args.facts)
    history: list[dict] = []
    rows = []
    t_start = time.time()
    for step in script:
        history.append({"role": "user", "content": step["text"]})
        if args.arm == "direct-full":
            ctx = history
        else:
            ctx = history[-args.window:]
        messages = [{"role": "system", "content": SYSTEM}] + ctx
        try:
            reply, usage, dt = chat(args.base, model, messages, args.max_tokens, args.timeout, extra)
            err = None
        except Exception as e:  # noqa: BLE001
            reply, usage, dt, err = "", {}, 0.0, str(e)[:200]
        history.append({"role": "assistant", "content": reply})
        hit = None
        if step["kind"] == "probe":
            norm = lambda s: s.lower().replace(",", "").replace(" ", "")
            hit = norm(step["value"]) in norm(reply)
        rows.append({**step, "reply": reply[:300], "hit": hit, "latency_s": round(dt, 2),
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "cost": usage.get("cost"),
                     "finish_reason": usage.get("finish_reason"), "error": err})
        print(f"t{step['turn']:3} {step['kind']:6} {dt:5.1f}s "
              f"{'HIT' if hit else ('MISS' if hit is False else '   ')} "
              f"{(usage.get('prompt_tokens') or 0):6} tok{'  ERR ' + err if err else ''}", flush=True)
    probes = [r for r in rows if r["kind"] == "probe"]
    lat = [r["latency_s"] for r in rows if not r["error"]]
    summary = {
        # host redacted: results are committed and endpoints are tailnet addresses
        "arm": args.arm, "model": args.model, "base": re.sub(r"(https?://)[^/]+", r"\1<host>", args.base),
        "window": args.window,
        "turns": len(rows), "probes": len(probes), "recall": sum(1 for r in probes if r["hit"]),
        "recall_rate": round(sum(1 for r in probes if r["hit"]) / max(1, len(probes)), 3),
        "mean_latency_s": round(sum(lat) / max(1, len(lat)), 2),
        "p90_latency_s": round(sorted(lat)[int(0.9 * len(lat)) - 1], 2) if lat else None,
        "total_s": round(time.time() - t_start, 1),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "cached_tokens": sum(r["cached_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        # Only present when the endpoint reports it (OpenRouter with usage.include).
        "cost_usd": round(sum(r["cost"] or 0 for r in rows), 4) if any(r.get("cost") is not None for r in rows) else None,
        "errors": sum(1 for r in rows if r["error"]),
        # A probe cut off by max_tokens is a measurement defect, not a miss;
        # surfaced so a low recall can be read correctly.
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
