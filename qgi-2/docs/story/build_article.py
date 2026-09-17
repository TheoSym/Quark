from build_story import *  # noqa: F401,F403

CSS = """
:root{--bg:#F4F5FA;--paper:#FFFFFF;--ink:#12162B;--ink2:#464C68;--ink3:#7A809C;--rule:#D9DCEA;--grid:#E7E9F3;--wash:#EAECF8;
--coral:#E4502F;--butter:#B07A00;--mint:#158F6B;--lilac:#6B55C9;--sky:#2B74D1;--butter-bg:#FFF3D1;
--h0:#EEF0F7;--h1:#D6E6FA;--h2:#A9CBF3;--h3:#6BA5E8;--h4:#F0A86B;--h5:#E4502F;
--gw:#E1F4EC;--ww:#FFF1CF;--bw:#FBE3DD;--hlo:#0E1430;
--disp:"Bricolage Grotesque","Arial Narrow",system-ui,sans-serif;--sans:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0E1430;--paper:#161E45;--ink:#F4F1EA;--ink2:#B8BEDA;--ink3:#8089B0;--rule:#2B3670;--grid:#232D63;--wash:#1B2554;
--coral:#FF6B4A;--butter:#FFD166;--mint:#4FD1A5;--lilac:#B9A7FF;--sky:#6FB3FF;--butter-bg:#33290F;
--h0:#18214D;--h1:#1D3E6B;--h2:#2A5C99;--h3:#3F82C9;--h4:#C9732F;--h5:#FF6B4A;--gw:#12332A;--ww:#33290F;--bw:#3A1B16;--hlo:#F4F1EA}}
:root[data-theme="dark"]{--bg:#0E1430;--paper:#161E45;--ink:#F4F1EA;--ink2:#B8BEDA;--ink3:#8089B0;--rule:#2B3670;--grid:#232D63;--wash:#1B2554;
--coral:#FF6B4A;--butter:#FFD166;--mint:#4FD1A5;--lilac:#B9A7FF;--sky:#6FB3FF;--butter-bg:#33290F;
--h0:#18214D;--h1:#1D3E6B;--h2:#2A5C99;--h3:#3F82C9;--h4:#C9732F;--h5:#FF6B4A;--gw:#12332A;--ww:#33290F;--bw:#3A1B16;--hlo:#F4F1EA}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:18px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 96px}
header.mast{padding:72px 0 36px;border-bottom:2px solid var(--ink)}
.kicker{font-family:var(--mono);font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--coral);margin:0 0 18px}
h1{font-family:var(--disp);font-weight:800;font-size:clamp(46px,8vw,104px);line-height:.94;letter-spacing:-.025em;margin:0 0 22px;text-wrap:balance}
.stand{font-size:23px;line-height:1.4;color:var(--ink2);max-width:60ch;margin:0}
.byline{font-family:var(--mono);font-size:12.5px;color:var(--ink3);margin-top:22px;letter-spacing:.04em}
a{color:var(--coral);text-underline-offset:3px}a:focus-visible{outline:2px solid var(--coral);outline-offset:2px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:18px;margin:34px 0 0}
.stat{border-top:3px solid var(--ink);padding-top:10px}.stat b{display:block;font-family:var(--disp);font-weight:800;font-size:34px;line-height:1}
.stat span{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink3)}
section{padding-top:64px}
.chap{font-family:var(--mono);font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--coral);margin:0 0 8px}
h2{font-family:var(--disp);font-weight:700;font-size:clamp(30px,4.4vw,52px);line-height:1.02;letter-spacing:-.018em;margin:0 0 18px;text-wrap:balance;max-width:20ch}
h3{font-family:var(--disp);font-weight:700;font-size:24px;margin:34px 0 8px}
.prose{max-width:66ch}.prose p{margin:0 0 18px;color:var(--ink2)}.prose p b{color:var(--ink);font-weight:600}
.prose p.first::first-letter{font-family:var(--disp);font-weight:800;font-size:4.1em;line-height:.8;float:left;padding:6px 10px 0 0;color:var(--coral)}
blockquote{margin:30px 0;padding:0 0 0 22px;border-left:5px solid var(--coral);font-family:var(--disp);font-weight:500;font-size:29px;line-height:1.18;letter-spacing:-.01em;max-width:24ch;color:var(--ink)}
code{font-family:var(--mono);font-size:.84em;background:var(--wash);padding:1px 6px;border-radius:4px}
figure{margin:26px 0;background:var(--paper);border:1px solid var(--rule);border-radius:12px;padding:20px 20px 12px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:14px;color:var(--ink3);margin-top:10px;max-width:84ch;line-height:1.45}
.ft{font-family:var(--sans);font-size:14px;font-weight:600;fill:var(--ink)}.fl{font-family:var(--sans);font-size:13px;fill:var(--ink2)}
.fn{font-family:var(--mono);font-size:12px;font-weight:500;fill:var(--ink)}.fa{font-family:var(--mono);font-size:11px;fill:var(--ink3)}.grid{stroke:var(--grid)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--ink2);margin:0 0 8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:22px 0}
.card{background:var(--paper);border:1px solid var(--rule);border-radius:12px;padding:18px 20px;border-top:5px solid var(--mint)}
.card.s{border-top-color:var(--sky)}.card h3{margin:0 0 8px;font-size:21px}.card ul{margin:0;padding-left:18px;color:var(--ink2);font-size:16px}.card li{margin:5px 0}.card li b{color:var(--ink)}
.tbl{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--paper);margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;line-height:1.4}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
.tbl td:first-child{white-space:nowrap}
tr:last-child td{border-bottom:0}
th{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--ink3);background:var(--wash);white-space:nowrap}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
table.sortable th{cursor:pointer;user-select:none}table.sortable th:hover{color:var(--coral)}
table.sortable th[aria-sort="ascending"]::after{content:" ▲";font-size:9px}table.sortable th[aria-sort="descending"]::after{content:" ▼";font-size:9px}
table.sortable th:focus-visible{outline:2px solid var(--coral);outline-offset:-2px}
.heat td.h{font-family:var(--mono);text-align:center;font-weight:500;min-width:96px}
.heat td.h span{display:block;font-size:10px;font-weight:400;opacity:.85}
.h0{background:var(--h0);color:var(--ink3)}.h1{background:var(--h1);color:var(--hlo)}.h2{background:var(--h2);color:var(--hlo)}.h3{background:var(--h3);color:#fff}.h4{background:var(--h4);color:#0E1430}.h5{background:var(--h5);color:#fff}
.chip{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:1px 8px;border-radius:999px;white-space:nowrap}
.chip.g{background:var(--gw);color:var(--mint)}.chip.w{background:var(--ww);color:var(--butter)}.chip.b{background:var(--bw);color:var(--coral)}
.bloopers{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin:22px 0}
.bloop{background:var(--paper);border:1px solid var(--rule);border-left:5px solid var(--butter);border-radius:0 12px 12px 0;padding:14px 18px;font-size:15.5px;color:var(--ink2)}
.bloop b{display:block;font-family:var(--mono);font-size:13px;color:var(--butter);letter-spacing:.04em;margin-bottom:4px}
.verdicts{counter-reset:v;display:grid;gap:12px;margin:22px 0;max-width:80ch}
.verdict{display:grid;grid-template-columns:56px 1fr;gap:14px;align-items:start;background:var(--paper);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}
.verdict::before{counter-increment:v;content:counter(v);font-family:var(--disp);font-weight:800;font-size:40px;line-height:.9;color:var(--coral)}
.verdict p{margin:0;color:var(--ink2);font-size:16.5px}.verdict p b{color:var(--ink)}
details{margin:14px 0;background:var(--paper);border:1px solid var(--rule);border-radius:10px}
summary{cursor:pointer;padding:12px 16px;font-family:var(--mono);font-size:13px;letter-spacing:.04em;color:var(--ink2)}
summary:hover{color:var(--coral)}details .tbl{border:0;border-top:1px solid var(--rule);border-radius:0 0 10px 10px;margin:0}
.hint{font-family:var(--mono);font-size:11.5px;color:var(--ink3);margin:4px 0 0}
footer{margin-top:72px;padding-top:20px;border-top:2px solid var(--ink);font-family:var(--mono);font-size:12px;color:var(--ink3);line-height:1.7}
@media (max-width:860px){.stats{grid-template-columns:repeat(3,1fr)}.two,.cards,.bloopers{grid-template-columns:1fr}body{font-size:17px}}
"""

ab_items = []
for label, d, q, td, tq, cache, wd, wq in AB[1:]:
    short = label.split(" · ")[0]
    ab_items.append(dict(label=f"pass {short} · jcode direct", v=wd, c="var(--sky)", note=d))
    ab_items.append(dict(label=f"pass {short} · through QGI-2", v=wq, c="var(--coral)" if wq > 20 else "var(--lilac)", note=q))
rec_items, tok_items = [], []
for m, arm, rec, tok, lat in MEM:
    short = ("DS V4.1" if m == V41 else "Flash-Next") + " · " + arm
    col = "var(--ink3)" if arm.startswith("whole") else ("var(--coral)" if arm.startswith("12") else "var(--lilac)")
    rec_items.append(dict(label=short, v=max(rec, 0.15), c=col, note=""))
    tok_items.append(dict(label=short, v=tok / 1000, c=col, note=f"{lat} s/turn"))
acc_items = [dict(label=l, v=hi, c="var(--coral)" if hi < 1.8 else "var(--mint)", note=(f"{lo}–{hi}" if lo != hi else "") + (" · " + n if n else "")) for l, lo, hi, n in ACCEPT]
lever_items = [dict(label=n, v=s1, c="var(--mint)" if n == "baseline" else "var(--ink3)", note=f"8 streams: {agg:.0f} · TTFT {t} s · accept {acc}") for n, t, s1, agg, acc in LEVERS]
m13 = []
for name, j, m, c, note in MATRIX13:
    m13 += [dict(label=f"{name} · jcode", v=j, c="var(--sky)"), dict(label=f"{name} · jcode + memory", v=m, c="var(--lilac)"), dict(label=f"{name} · codex", v=c, c="var(--butter)", note=("1/3" if "1/3" in note else ""))]
doc_items = [dict(label=n, v=mins, c=("var(--mint)" if "self-hosted" in where or "B200" in where else "var(--sky)"), note=f"{j} tasks · {ch} checks") for n, where, j, ch, mins, mj, mch, note in DOC]

def modes_bars(key, name):
    items = []
    for a in ARMS:
        c = CELLS.get((key, a))
        items.append(dict(label=ARM_LABEL[a], v=(c["median"] if c else None), c=("var(--lilac)" if a.startswith("mem") else "var(--sky)"),
                          note=("" if c else "not run"), tip=(f"{name}: {c['in'] // 1000}K in · {c['out']} out" if c else name)))
    return hbars(items, title=name, label_w=150, bar_w=230, row=30, note_w=70)

# ---------------- tables
gate_rows = []
for g in sorted(GATE, key=lambda g: (-(g["v"]["correctness_rate"] or 0), g["v"]["coding_median_wall_s"] or 9e9)):
    v = g["v"]
    gate_rows.append([gate_name(g), g["route"], chip(v["correctness"]), v["coding"], v["agentic"], v["memory"], v["coding_median_wall_s"], v["decode_tps"], v["ttft_s"],
                      v.get("tokens_per_task"), (f"{v['cost_per_task_usd']:.4f}" if v.get("cost_per_task_usd") is not None else None),
                      (f"{v['gate_cost_usd']:.3f}" if v.get("gate_cost_usd") is not None else None), (round(g["wall"]) if g["wall"] else None), v["errors"]])
task_rows = [[gate_name(dict(model=r["model"], effort=r["effort"])), r["leg"], r["task"], chip(r["solved"]) if r["solved"] and "/" in str(r["solved"]) else (r["solved"] or "—"),
              r["wall"], r["inp"], r["out"], r["cached"], (f"{r['cost']:.4f}" if (r["cost"] is not None and "/" in r["model"]) else None), r["note"]] for r in TASKS]
mode_rows = []
for key, name, route in MODELS:
    for a in ARMS:
        c = CELLS.get((key, a))
        if not c:
            mode_rows.append([name, route, ARM_LABEL[a], "not run", None, None, None, None, None, None, None, None, None])
            continue
        cc = cell_cost(key, c)
        per = [(f"{c['per'][e][0]}" + ("" if c["per"][e][1] else " ✗")) if e in c["per"] else None for e in ("bowling", "forth", "wordy")]
        mode_rows.append([name, route, ARM_LABEL[a], chip(f"{c['solved']}/{c['n']}"), c["median"], c["total"], c["in"], c["out"],
                          (f"{100 * (c['cached'] or 0) / c['in']:.0f}%" if c.get("in") else None), (f"{cc:.4f}" if cc is not None else None),
                          (f"{cc / c['n']:.4f}" if cc is not None else None)] + per[:2] + [per[2]])
mode_rows = [r[:13] + ([r[13]] if len(r) > 13 else []) for r in mode_rows]

H = ["<title>Four Days on Rented Silicon</title>", FONTS, "<style>" + CSS + "</style>", '<div class="wrap">']
H.append(f"""<header class="mast"><p class="kicker">A field report from the QGI-2 and Quark trials · 11–17 September 2026</p>
<h1>Four Days on Rented Silicon</h1>
<p class="stand">We rented a supercomputer that couldn't count to three, built a planner that lost to the thing it was planning for, taught a proxy to remember a Wi-Fi password from 88 turns ago, graded sixteen models on a test they nearly all aced, and ran a credit card down to $1.85. Here is everything we measured, and what it means.</p>
<p class="byline">Starring {V41} and {QFN} · with Claude Fable 5.1, Claude Opus 5, Mercury 2.5, Grok 4.6, GPT-6 Astra and nine gateway regulars</p>
<div class="stats"><div class="stat"><b>2</b><span>GPU boxes · one lost</span></div><div class="stat"><b>24</b><span>models touched</span></div><div class="stat"><b>402</b><span>harness tests</span></div>
<div class="stat"><b>240</b><span>gate verdicts</span></div><div class="stat"><b>48</b><span>matrix cells, all 3/3</span></div><div class="stat"><b>$122.79</b><span>OpenRouter, two days</span></div></div></header>""")

H.append(f"""<section><p class="chap">Chapter 1 · the premise</p><h2>A spec, a harness, and a rule: don't touch jcode</h2><div class="prose">
<p class="first">The spec asked for something specific. Two models, a planner and a worker. A prompt cut into six segments, the first three byte-stable so the engine's prefix cache actually hits. Typed facts that the model may only <i>propose</i>, with rules deciding what gets committed. And every single step declared as a triple of model, speculation and sampling, because "nothing defaults".</p>
<p>QGI-2 is what came out: eleven crates sitting beside jcode with <b>zero edits to jcode itself</b>, an engine trait that speaks vLLM and SGLang, HiCache tiers and a calculator for the recurrent-state pool that hybrid models quietly need, two edges (an HTTP sidecar and an in-process provider), and a scripted engine that makes the whole agent loop testable without a GPU. The test count ended at <b>402</b>. A Datalog engine went in and came back out after an audit found fifteen rules of which exactly one was recursive; a worklist does that job in ten lines.</p>
<p>Then there is <b>Quark</b>: jcode, branded and wired for the app factory. A stdlib Python CLI with its own home directory and hub profile, a runner that translates jcode's event stream into the app's AG-UI, a prompt that weighs 9K tokens instead of 33K because nothing from a personal config leaks in, and exactly one prompt policy on board, which stays off unless you ask for it. 47 of 47 runner tests pass.</p></div></section>""")

H.append(f"""<section><p class="chap">Chapter 2 · infrastructure</p><h2>The box that couldn't count to three</h2><div class="prose">
<p class="first">The first rental was eight RTX PRO 6000 cards. CUDA initialised happily on any one GPU. On any two. On any three, it failed, every time, with a UVM error that turned out to be a known host driver bug needing a kernel module reload nobody renting the machine can perform. SM120 also wanted <code>FLASHINFER_CUDA_ARCH_LIST=12.0a</code> or FlashInfer's probe tripped the same bug. The box was later destroyed outright by a failed VM conversion, which at least settled the question.</p>
<p>Its replacement was four B200s at 183 GB each, <b>$28 an hour</b> whether serving or idle. The 510 GB planner checkpoint arrived at 5 GB/s. First launch took <b>35 minutes</b> of kernel JIT and autotune; later launches, five. The obvious layout, tensor-parallel across all four cards, was mostly empty space: a 39-million-token KV pool nobody needed and 556 GB of host RAM pinned for a cache tier that could never get a hit. Two cards at a 0.92 memory fraction with the pool capped at four million tokens served the same model with DSpark accepting 2.9 tokens a step. Three cards is not an option: the vocabulary is 129,280 entries and that does not divide by three.</p></div>
{tbl(["card", "tenant", "memory", "speculation"], [["0 + 1", V41 + " (510 GB planner, TP2, EP2)", "166 GB each", "DSpark, block 5, HiCache ratio 2"], ["2", QFN + " (RadixArk NVFP4 export)", "163 GB, fraction 0.85", "NEXTN 3/1/4, fp8 KV, 3.0 M-token pool"], ["3", "Qwen3.8-27B FP8 + DFlash2 worker", "fraction 0.30", "DFLASH, 8 drafts"], ["3", "Qwen3.6-35B-A3B FP8 + DFlash", "fraction 0.55 (0.45 fails: hybrid state pool)", "DFLASH, 8 drafts"], ["3", "DiffusionGemma 26B NVFP4 on vLLM", "fraction 0.17", "none"]], cls="")}
<div class="prose"><p>Before the meter was stopped for good, everything needed to rebuild the box went into the repo: 488 files of launch scripts, supervisor units, both SGLang commits with the one uncommitted patch as a diff, venv freezes, model manifests and the last 4,000 lines of every engine log, plus a 209 MB tarball of JIT caches that turns a 35-minute first boot into five.</p></div></section>""")

H.append(f"""<section><p class="chap">Chapter 3 · the locals</p><h2>Meet the two models that held the line</h2>
<div class="cards"><div class="card"><h3>{V41}</h3><ul><li><b>254 tok/s</b> single stream, <b>1,337 tok/s</b> across eight</li><li>TTFT <b>0.18 s</b> on a 7.7K-token prompt</li><li>DSpark accepts 2.6–3.9 tokens per step; floor is 1.8</li><li>14.8 s per Exercism exercise, 98 % cache</li><li>Documents: 7/7, 58/58 checks, 9.6 minutes</li><li>Prefix cache ceiling of 74–76 % on short prompts: the backend fixes pages at 256 tokens</li></ul></div>
<div class="card s"><h3>{QFN}</h3><ul><li>MTP accept <b>1.2–1.7 → 3.2–3.75</b> by changing only the NVFP4 export</li><li>157 tok/s with MTP, 144–155 without</li><li>Polyglot <b>16/16</b>, 16.0 s median; codex on the same card 33.3 s</li><li>fp8 KV, 3,048,512-token pool, 327 recurrent-state slots</li><li>Best arm: safe-efficient, 26.0 → 15.2 s</li><li>Needs SGLang main; the planner's build has no class for it</li></ul></div></div>
<figure>{hbars(acc_items, title="Speculative decoding: accepted tokens per step, upper end of the measured range", label_w=400, bar_w=300, vmax=7, row=32, note_w=400)}<figcaption>The spec sets floors of 1.8 for the planner and 2.0 for the worker. One configuration sits under a floor, and it is the same checkpoint on the same card with the same flags as the one above it: the export decided. The spec's own table also said DFlash2 "can't do greedy". It does, at 4.5–5.4.</figcaption></figure>
<figure>{hbars(lever_items, title=V41 + " · four speed levers, none adopted · single-stream tok/s", label_w=200, bar_w=300, vmax=300, row=32, note_w=400)}<figcaption>Q8KV8 sparse prefill and DeepEP each came in a few percent slower; the fourth lever, fused-greedy DSpark, left no result in the log.</figcaption></figure></section>""")

H.append(f"""<section><p class="chap">Chapter 4 · the harness</p><h2>We built a planner. jcode beat it by 28 seconds.</h2><div class="prose">
<p class="first">The honest test of a harness is an A/B against the thing it wraps: same model, same tasks, stock jcode talking straight to the engine versus stock jcode routed through QGI-2. The first pass went 4/4 against 1/4, and every one of the harness's failures was ours. The edge answered a streaming request with a JSON body, so jcode saw silence. The planner was never told the real tool names and invented <code>fs.read</code>. Every run shared one session, so one task's <code>calc.py</code> facts leaked into the next. And stored tool output was sentence-split, so the edit after a read could never match bytes and the planner re-read the same file five rounds running.</p>
<p>Seven fixes later, pass two was the honest one: <b>6/8 at 33.8 seconds</b> against 8/8 at 5.8. Plan, then tool arguments, then extraction is three engine calls a round where jcode spends one, and a planner separated from its tools is a worse tool user than a model calling them natively.</p></div>
<blockquote>The harness's value was never speed. It was memory.</blockquote>
<div class="prose"><p>So pass three threw the planner out. jcode stays native for everything; QGI-2 becomes a pass-through proxy that records the transcript into a line store, retrieves what matters, and inserts a single <code>&lt;memory&gt;</code> message before the latest user turn, frozen for the turn so the cached prefix stays byte-identical. <b>8/8, 5.4 seconds, four percent more tokens.</b> Pass four added the spaCy ingester and per-turn compaction and paid one second for them.</p></div>
<figure>{hbars(ab_items, title="Median seconds per task, direct versus through QGI-2", label_w=300, bar_w=460, vmax=36, row=30)}</figure>
{tbl(["pass", "jcode direct", "through QGI-2", "tokens direct", "tokens QGI-2", "cache", "wall direct s", "wall QGI-2 s"], [[l, chip(d), chip(q), td, tq, c, wd, wq] for l, d, q, td, tq, c, wd, wq in AB], num_cols=(3, 4, 6, 7))}</section>""")

H.append(f"""<section><p class="chap">Chapter 5 · memory</p><h2>Eighty-eight turns later, it still knows the Wi-Fi password</h2><div class="prose">
<p class="first">Coding tasks never stress memory; they finish long before any budget is hit. So the memory test is a hundred-turn conversation at a fictional robotics startup: twenty facts planted early (the gripper's torque rating, the test floor's SSID, the finance lead's name), filler in between, twenty probes late. Real clients truncate history, so the model only sees the last twelve turns.</p>
<p>With the whole transcript in context, both local models recall 20 of 20, at 169K and 217K prompt tokens. With the window alone: <b>zero of twenty</b>. With the window plus the proxy: <b>20/20</b> on {V41} and <b>18/20</b> on {QFN}, at 75K and 85K tokens, which is 44 % and 39 % of the whole-transcript bill, and faster per turn because the prompts are shorter. That morning's first version had scored 13 and 10; the ingester and the compactor closed the gap, and no model changed.</p></div>
<div class="two"><figure>{hbars(rec_items, title="Facts recalled, of 20", label_w=215, bar_w=190, vmax=22, row=34, fmt="{:.0f}", note_w=50)}</figure><figure>{hbars(tok_items, title="Prompt tokens over the run, thousands", label_w=215, bar_w=130, vmax=240, row=34, fmt="{:.0f}K", note_w=130)}</figure></div></section>""")

H.append(f"""<section><p class="chap">Chapter 6 · office hours</p><h2>Seven documents, fifty-eight checks, and one model that only talks about files</h2><div class="prose">
<p class="first">Agents that can code are common; agents that can produce a correct invoice PDF are rarer. The document bench asks for a Word report, an HTML data report, a memo exported to PDF, a flyer with logo and deck, a PDF invoice, a six-slide deck and an Excel model. Verifiers open the files and check them. A task passes only if every check passes.</p>
<p>{V41} went <b>7 for 7, 58 of 58 checks, in 9.6 minutes</b>. Gemini 3.8 Flash matched it in 12.3. The gateway's own DeepSeek V4 Pro also got there, in <b>74 minutes</b> with six gateway errors. DiffusionGemma cheerfully described the file it was about to create and mostly did not create it. The memory proxy cost roughly one task per model here: retrieved lines occasionally steered a run toward an earlier task's file. That is the price, on short sessions, of chapter five.</p></div>
<figure>{hbars(doc_items, title="Minutes to finish all seven tasks (jcode arm)", label_w=330, bar_w=360, vmax=80, row=30, note_w=300)}</figure>
{tbl(["model", "where", "jcode", "checks", "minutes", "+ memory", "checks", "note"], [[n, w, chip(j), ch, mins, chip(mj), mch, note] for n, w, j, ch, mins, mj, mch, note in DOC], num_cols=(4,))}
<h3>The coding matrix of 13 September</h3>
<figure>{hbars(m13, title="Median seconds per exercise · three agents per model", label_w=380, bar_w=380, vmax=150, row=24)}<figcaption>codex reports 0 % cache on every self-hosted model: it speaks the Responses API, which needed a 26-line backport before it worked at all.</figcaption></figure></section>""")

H.append(f"""<section><p class="chap">Chapter 7 · the gate</p><h2>Everyone gets an A. Speed does the grading.</h2><div class="prose">
<p class="first">All of that became one command. The gate runs four legs against one model in ten to twenty minutes: an engine probe, three Exercism exercises, two document tasks plus a two-turn task whose second turn depends on the first, and eight facts recalled from a 40-turn context. Fifteen objective verdicts, nothing judged by another model.</p>
<p>Sixteen models took it. <b>Fourteen scored 15/15.</b> Claude Opus 5 at max effort dropped two, both recall questions answered with an empty string and <code>finish_reason: content_filter</code>. The <code>:batch</code> variant of Fable turned out to be batch-API-only and returned fifty 404s in eleven seconds. QGI-Flash 3.8 Next answered nothing in six ten-minute tasks; the endpoint was simply down. With capability a tie, the clock decides: the two locals at 14.8 and 17.0 seconds, Fable at low effort at 24.3, then a pack of gateway models between 50 and 77, then the reasoners, and finally the gateway's DeepSeek pinned against the ten-minute cap, passing its tests and then refusing to stop.</p></div>
<div class="legend"><span><i style="background:var(--mint)"></i>self-hosted, 13 Sep reference</span><span><i style="background:var(--coral)"></i>direct on OpenRouter</span><span><i style="background:var(--sky)"></i>through the gateway</span></div>
<figure>{hbars(LADDER, title="Median seconds per coding task", label_w=360, bar_w=400, vmax=240, row=27, note_w=280)}<figcaption>Bars clip at 240 s; DeepSeek V4 Pro's true figure is 600.9, the per-task cap. Mercury 2.5 decodes at 342 tok/s, the fastest anything measured here, and one of its three exercises still ran to the cap after passing.</figcaption></figure>
<p class="hint">click a column header to sort · click again to reverse</p>
{tbl(["model", "route", "correct", "coding", "agentic", "memory", "s/task", "tok/s", "TTFT s", "tok/task", "$/task floor", "gate $ floor", "gate wall s", "errors"], gate_rows, num_cols=(6, 7, 8, 9, 10, 11, 12, 13))}</section>""")

mem_fig = hbars(MEMRATIO, title="Memory arm ÷ baseline, median seconds", label_w=300, bar_w=420, vmax=5.2, row=32, fmt="{:.2f}×")
H.append(f"""<section><p class="chap">Chapter 8 · the matrix</p><h2>Prompt diets help some. The memory tax hits others.</h2><div class="prose">
<p class="first">Eight models, six arms, three exercises, forty-eight cells. Four arms change only the prompt jcode runs under: baseline, a safe-efficient coding policy, a search-first "exa-reuse" policy, and the same with the Exa search server actually connected. Two arms put the QGI-2 memory proxy in the path, alone and together with the Exa arm. The two local rows are the 13 September runs, carried over unchanged.</p>
<p><b>Every cell solved 3 of 3.</b> The arms move seconds and tokens and nothing else, and their sign depends on the model. The safe-efficient diet speeds up Gemini (44 → 34 s), Opus, GLM and both locals, and slows Fable and QGI 3.8 Flash. The memory proxy is free on Fable, where it is the fastest arm of six, a gift to QGI-V4 Pro, neutral on Opus, and a tax on Gemini and above all GLM 5.3, where it costs <b>4.8×</b>. On Gemini the combination of memory and tool schema drops the cache ratio from 90 % to 59 %: the first measured case of QGI-2's own additions defeating its cache discipline.</p>
<p>And Exa? jcode connected the search server with two tools on all 48 tool-arm runs. <b>No model called it once.</b> On the 13th the local models did, eight times in 56 runs. The tool arms measure the price of offering search, not of using it.</p></div>
<p class="hint">★ fastest arm per model · click a header to sort</p><div class="tbl">{modes_heat()}</div>
<figure>{mem_fig}<figcaption>Green within 5 % of baseline, amber to 1.6×, coral beyond. Status colour here is the verdict, not a series.</figcaption></figure>
<div class="two">{"".join("<figure>" + modes_bars(k, n) + "</figure>" for k, n, r in MODELS)}</div></section>""")

H.append(f"""<section><p class="chap">Chapter 9 · the bill</p><h2>What the agent reports and what the card is charged</h2><div class="prose">
<p class="first">What jcode reports is input, output and cache-read tokens. What it does not report is reasoning tokens, and reasoning is what a high-effort model spends most on. So the priced cost of the six frontier gates, computed honestly from the rate card and jcode's counts, came to <b>$5.83</b>. The card was charged <b>$46.99</b>. GPT-6 Astra shows 40 to 61 output tokens per exercise and takes three minutes over each one; the difference is thinking you pay for and never see counted.</p>
<p>Halfway through the matrix every call began failing with HTTP 402: the account could "only afford 62,583" tokens, with <b>$1.85</b> left of 335. The surprise was the blast radius. The gateway routes Gemini, QGI-V4 Pro, QGI 3.8 Flash and GLM 5.3 through the very same OpenRouter account, so the "gateway" sweep that afternoon had been drawing it down all along: $63.87 on the 14th. After a top-up the 48-cell matrix cost $58.92. Two days, <b>$122.79</b>. Mercury 2.5's entire 15-verdict gate, for scale: one cent.</p></div>
<div class="legend"><span><i style="background:var(--mint)"></i>self-hosted, the cards' share of the $28/h box</span><span><i style="background:var(--coral)"></i>OpenRouter rate card</span><span><i style="background:var(--sky)"></i>gateway rate card</span></div>
<figure>{hbars(COST, title="USD per exercise, baseline arm", label_w=330, bar_w=360, row=30, fmt="${:.4f}", note_w=330)}<figcaption>Output tokens are under 1,000 for three whole exercises on every model. What you pay for is context, which is why every overlay and tool schema shows up in the bill.</figcaption></figure></section>""")

H.append('<section><p class="chap">Chapter 10 · blooper reel</p><h2>Ten things that went wrong, in ascending order of embarrassment</h2><div class="bloopers">'
         + "".join(f'<div class="bloop"><b>{esc(t)}</b>{esc(x)}</div>' for t, x in BLOOPERS) + "</div></section>")

H.append(f"""<section><p class="chap">Chapter 11 · what we keep</p><h2>Remember more, plan less, host your own</h2><div class="verdicts">
<div class="verdict"><p><b>The memory layer stays.</b> Zero to twenty on the long-horizon test, parity on coding, free on the models that matter. Tune it for the models it taxes: GLM 5.3 and Gemini first.</p></div>
<div class="verdict"><p><b>The step-split planner goes.</b> Three engine calls a round lost to one, and a planner without its tools invents them.</p></div>
<div class="verdict"><p><b>Self-hosting wins on speed.</b> 14.8 seconds against 24.3 for the best frontier model and 50 and up through the gateway, with document work at parity with Gemini.</p></div>
<div class="verdict"><p><b>Effort is a lever, not a virtue.</b> Max and high effort bought minutes, not answers, on tasks this size.</p></div>
<div class="verdict"><p><b>Prompt policy is per model.</b> Which is why safe-efficient ships in Quark as an opt-in mode and not a default.</p></div>
<div class="verdict"><p><b>Count the money at the card, not the counter.</b> The gate now records the key's spend delta, because reasoning tokens hide from the agent.</p></div>
<div class="verdict"><p><b>Quark ships.</b> CLI and runner in the app factory, its own home, a 9K-token prompt, 47 of 47 tests.</p></div></div></section>""")

H.append(f"""<section><p class="chap">Appendix · every number</p><h2>The ledgers</h2>
<details open><summary>Coding-modes matrix · 8 models × 6 arms · every cell ({len(mode_rows)} rows)</summary>{tbl(["model", "route", "arm", "solved", "median s", "total s", "input tok", "output tok", "cache", "$ arm", "$/task", "bowling", "forth", "wordy"], mode_rows, num_cols=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13))}</details>
<details><summary>Gate, task by task · 16 models × 9 rows ({len(task_rows)} rows)</summary>{tbl(["model", "leg", "task", "result", "wall s", "input tok", "output tok", "cache-read tok", "$ floor", "note"], task_rows, num_cols=(4, 5, 6, 7, 8))}</details>
<details><summary>Long-horizon memory eval · 100 turns, 20 probes, 12-turn window</summary>{tbl(["model", "arm", "recall /20", "prompt tokens", "mean s per turn"], [[m, a, r, t, l] for m, a, r, t, l in MEM], num_cols=(2, 3, 4))}</details>
<details><summary>Planner speed levers · {V41}</summary>{tbl(["lever", "TTFT s", "single-stream tok/s", "8-stream aggregate tok/s", "DSpark accept"], [list(x) for x in LEVERS], num_cols=(1, 2, 3, 4))}</details>
<details><summary>13 September model × agent matrix</summary>{tbl(["model", "jcode s", "jcode + memory s", "codex s", "note"], [list(x) for x in MATRIX13], num_cols=(1, 2, 3))}</details>
<details><summary>13 September prompt policies on the two locals</summary>{tbl(["model", "baseline", "safe-efficient", "exa-reuse", "exa-reuse + tool"], [[n] + [CELLS[(k, a)]["median"] for a in ARMS[:4]] for k, n, r in MODELS[:2]], num_cols=(1, 2, 3, 4))}</details>
<details><summary>Infrastructure lessons that cost real hours</summary>{tbl(["finding", "consequence"], [
["8× RTX PRO 6000, driver 590.48: CUDA init fails for any 3+ GPUs", "Host-side nvidia_uvm reload is the only fix; test CUDA_VISIBLE_DEVICES=0,1,2 before keeping a Blackwell rental."],
["SM120 needs FLASHINFER_CUDA_ARCH_LIST=12.0a; B200 must not set it", "FlashInfer's probe enumerates every GPU and trips the bug; SM100 auto-detects correctly."],
["First launch 35 min (DS V4.1), 15 min (Flash-Next)", "Caches survive stop/start, not recycle; the 209 MB tarball restores a 5-minute restart."],
["SGLang main has no V4.1 classes; the dsv41 build has no Flash-Next classes", "Two venvs are mandatory; /v1/responses needed a token-first prompt backport for codex."],
["--enable-cache-report absent", "cached_tokens reads 0 on every reply and the harness flags a false breach."],
["Gateway reasoning tokens count against max_tokens", "Use 2048 or more; a 120 cap truncated recall answers."],
["The gateway's online models bill one OpenRouter account", "A credit wall there takes down direct and gateway runs together."]], cls="")}</details>
</section>""")

H.append(f"""<footer>Sources: Quark/qgi-2/bench/results (gate, modes-2026-09-14, coding-modes-2026-09-13, docbench-2026-09-13, memory-eval-v3-2026-09-13, matrix-2026-09-13) · qgi-2/deploy/V41-BRINGUP.md · deploy/snapshots/b200-50681041 · QGI-apps tools/quark.<br>
Model names as billed for this piece: "{V41}" is the self-hosted DeepSeek-V4.1-Flash deployment (TP2, DSpark, HiCache) and "{QFN}" the self-hosted Qwen3.8-Flash-Next NVFP4 deployment (RadixArk export, MTP, fp8 KV). Token-priced costs are floors; one run per cell; an entertainment, not a paper.</footer></div>""")
H.append("<script>(function(){var t=new URLSearchParams(location.search).get('theme');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);})();</script>")
H.append(SORT_JS)
open(os.path.join(OUT, "quark-story-article.html"), "w", encoding="utf-8").write("\n".join(H))
print("article written", sum(len(x) for x in H))
