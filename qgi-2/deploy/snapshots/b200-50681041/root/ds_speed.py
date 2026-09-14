#!/usr/bin/env python3
"""DeepSeek planner speed probe, same prompts every run:
prefill TTFT on a ~4k-token prompt (3 reps, cache-defeating suffix), single-stream
decode tok/s (3 reps), 8-stream aggregate decode tok/s. Usage: ds_speed.py [base] [model]"""
import sys, json, time, urllib.request, concurrent.futures as cf

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18034"
M = sys.argv[2] if len(sys.argv) > 2 else "QGI-2-V41"
LONG = ("def f(x):\n    return x * 2\n" * 700) + "\nSummarize what the code above does in one sentence."


def call(prompt, max_tokens, stream=False):
    body = {"model": M, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "low", "stream": stream}
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        return time.time() - t0, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"], None
    first = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                if first is None:
                    first = time.time() - t0
                n += 1
    return time.time() - t0, n, None, first


call("warmup", 20)
tt = []
for i in range(3):
    _, _, _, first = call(LONG + f" (run {i} {time.time()})", 64, stream=True)
    tt.append(first)
_, _, ptok, _ = call(LONG + " (count)", 1)
print(f"prefill: {ptok} tok prompt, TTFT median {sorted(tt)[1]:.2f}s  (runs {', '.join(f'{x:.2f}' for x in tt)})")
ds = []
for i in range(3):
    dt, n, _, _ = call(f"Write a detailed 400-word essay about the history of bridges. Variant {i}.", 400)
    ds.append(n / dt)
print(f"decode 1-stream: median {sorted(ds)[1]:.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in ds)})")
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex:
    rs = list(ex.map(lambda i: call(f"Write a detailed 300-word essay about topic number {i}.", 300), range(8)))
tot = sum(r[1] for r in rs)
print(f"decode 8-stream: {tot / (time.time() - t0):.1f} tok/s aggregate, {tot} tokens in {time.time() - t0:.1f}s")
try:
    m = urllib.request.urlopen(B + "/metrics", timeout=10).read().decode()
    for line in m.splitlines():
        if line.startswith("sglang:spec_accept_length"):
            print("accept_len", line.rsplit(" ", 1)[-1])
except Exception as e:  # noqa: BLE001
    print("metrics unavailable:", e)
