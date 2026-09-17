"""Build the 10-slide deck and the long-form article from the real result files."""
import glob
import html
import json
import os
import sys

sys.path.insert(0, "F:/QHG/Quark/qgi-2/bench")
import modes_report as mr  # noqa: E402

R = "F:/QHG/Quark/qgi-2/bench/results"
OUT = sys.argv[1]
V41 = "Enhanced QGI DS V4.1-Flash local"
QFN = "QGI Enhanced Qwen Flash-Next"
esc = html.escape

# ------------------------------------------------------------------ data
GATE = []
for f in sorted(glob.glob(R + "/gate/gate-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    v = d["verdict"]
    route = "OpenRouter" if "/" in d["model"] else "gateway"  # endpoints are redacted in the files
    GATE.append(dict(model=d["model"], effort=d.get("reasoning_effort") or "", route=route, v=v, wall=d.get("gate_wall_s")))
TASKS = json.load(open(R + "/gate/tasks-2026-09-14.json", encoding="utf-8"))
CELLS = mr.load(R + "/modes-2026-09-14")
ARMS = mr.ARMS
ARM_LABEL = {"baseline": "baseline", "safe-efficient": "safe-efficient", "exa-reuse": "exa-reuse",
             "exa-reuse+tool": "exa-reuse + tool", "mem": "memory proxy", "mem+exa-tool": "memory + exa-tool"}
MODELS = [("ds41", V41, "local"), ("qwen38", QFN, "local"), ("fable-low", "Claude Fable 5.1 · low", "openrouter"),
          ("opus-max", "Claude Opus 5 · max", "openrouter"), ("gem", "Gemini 3.8 Flash", "gateway"),
          ("v4pro", "QGI-V4 Pro", "gateway"), ("q38", "QGI 3.8 Flash", "gateway"), ("glm53", "GLM 5.3", "gateway")]
RATES = {  # USD per token: prompt, completion, cache read
    "fable-low": (10e-6, 50e-6, 0.25e-6), "opus-max": (5e-6, 25e-6, 0.5e-6),
    "gem": (0.75e-6, 3.75e-6, 0.075e-6), "v4pro": (0.57948e-6, 1.73844e-6, 0.019316e-6),
    "q38": (0.15e-6, 0.47e-6, 0.016e-6), "glm53": (1.4e-6, 4.4e-6, 0.14e-6)}


def cell_cost(key, c):
    if key in ("ds41", "qwen38"):
        # the cards each model occupied: 2 of 4 for DS V4.1, 1 of 4 for Flash-Next, of a $28/h box
        return (14.0 if key == "ds41" else 7.0) / 3600 * (c.get("total") or 0)
    r = RATES.get(key)
    if not r or c.get("in") is None:
        return None
    i, o, k = c["in"] or 0, c["out"] or 0, c["cached"] or 0
    return (i - k) * r[0] + k * r[2] + o * r[1]


AB = [("1 · step-split, first wiring", "4/4", "1/4", None, None, None, None, None),
      ("2 · step-split, after 7 fixes", "8/8", "6/8", 170351, 218945, "98 → 50 %", 5.8, 33.8),
      ("3 · memory-only proxy", "8/8", "8/8", 168092, 174168, "98 → 91 %", 5.5, 5.4),
      ("4 · proxy + spaCy ingester + every-turn compaction", "8/8", "8/8", 235727, 236845, "99 → 96 %", 6.6, 7.7)]
MEM = [(V41, "whole transcript", 20, 169034, 0.42), (V41, "12-turn window", 0, 27230, 0.44), (V41, "window + memory proxy", 20, 74947, 0.32),
       (QFN, "whole transcript", 20, 217033, 0.41), (QFN, "12-turn window", 0, 31176, 0.42), (QFN, "window + memory proxy", 18, 85291, 0.29)]
MATRIX13 = [(V41, 14.8, 46.4, 14.1, ""), (QFN, 17.0, 29.3, 33.6, "full 16-exercise set: jcode 16/16 at 16.0 s, codex 16/16 at 33.3 s"),
            ("Qwen3.8-27B + DFlash2 (worker)", 56.9, 50.9, 75.4, "0.30 of a shared card"),
            ("QGI 3.8 Flash (gateway)", 21.6, 28.4, 58.1, ""), ("Muse-Glimmer-30B NVFP4", 63.9, 143.3, 44.6, "codex solved 1/3; dropped from the fleet")]
DOC = [(V41, "B200, self-hosted", "7/7", "58/58", 9.6, "6/7", "57/58", "one check short with memory on"),
       ("Gemini 3.8 Flash", "gateway", "7/7", "58/58", 12.3, "6/7", "57/58", ""),
       ("DeepSeek V4 Pro", "gateway", "7/7", "58/58", 74.3, "5/7", "47/51", "6 gateway errors, two 900 s timeouts"),
       ("Qwen3.8-27B + DFlash2", "B200 card 3", "6/7", "48/51", 38.8, "6/7", "48/51", ""),
       ("Qwen3.6-35B-A3B + DFlash", "B200 card 3", "6/7", "48/51", 38.6, "4/7", "46/51", "two timeouts with memory on"),
       (QFN, "B200 card 2", "4/7", "49/52", 36.0, "6/7", "48/51", "one 'fail' was the verifier crashing"),
       ("DiffusionGemma 26B NVFP4", "B200 card 3, vLLM", "1/7", "18/26", 6.0, "1/5", "10/16", "talks about the file, rarely writes it; agentd arm 0/7")]
LEVERS = [("baseline", 0.18, 254.2, 1336.8, 2.85), ("Q8KV8 sparse prefill", 0.21, 247.5, 1230.6, 2.81), ("DeepEP all-to-all", 0.21, 246.9, 1198.8, 2.68)]
ACCEPT = [(QFN + " · nvidia export · MTP", 1.2, 1.7, "under the 1.8 floor"), (QFN + " · RadixArk export · MTP", 3.2, 3.75, "accept rate 0.74–0.92"),
          (V41 + " · DSpark block 5", 2.6, 3.9, "2.85 in the lever sweep"), ("27B worker · DFlash2 · greedy", 4.5, 5.4, "the spec said this was impossible"),
          ("27B worker · DFlash2 · sampled", 6.45, 6.45, "accept rate 0.78")]
BLOOPERS = [
    ("HR-229", "A recall probe asked for ticket HR-2291 and got \u201cHR-229\u201d. The gateway counts reasoning tokens against max_tokens; 113 of a 120-token cap went to thinking. Raised to 2048; Gemini went from 4/8 to 8/8."),
    ("The curly quote", "A model emitted one typographic quote, Windows decoded it as cp1252, and the whole stateful run fell over. Children now run under UTF-8."),
    ("Wrong key, right door", "The gate handed OpenRouter the LiteLLM master key. 401 on every direct leg, in 0.3 seconds flat."),
    ("content_filter", "Opus at max effort returned two empty answers to \u201cwhat is the name of our lead firmware engineer\u201d with finish_reason content_filter. Its only two misses."),
    ("The batch that wasn't", "anthropic/claude-fable-5.1:batch answers 404: \u201conly available through the Batch API\u201d. Fifty errors in eleven seconds."),
    ("Finish, then keep going", "Seven models passed their tests and kept working until the 600 s cap. Correct, and expensive in wall clock."),
    ("The 402 wall", "Mid-matrix, every call started failing: the account could \u201conly afford 62,583\u201d tokens. $1.85 left. Surprise: the gateway routes its online models through the same OpenRouter account."),
    ("A GPU box that can't count to three", "8\u00d7 RTX PRO 6000: CUDA initialises for any 1 or 2 GPUs and fails for any 3+. Host driver bug; the box was later destroyed by a failed VM conversion."),
    ("Nobody called Exa", "48 tool-arm runs, Exa connected every time, zero calls. On the 13th the local models called it 8 times in 56 runs."),
    ("DFlash2 can't do greedy", "The spec said so. Measured: 4.5\u20135.4 accepted tokens per step, greedy. The veto was deleted."),
]

# ------------------------------------------------------------------ svg helpers
def hbars(items, title=None, label_w=300, bar_w=520, row=30, vmax=None, fmt="{:.1f}", unit="", ticks=4, aria="", note_w=190):
    vmax = vmax or max(i["v"] for i in items if i["v"] is not None) * 1.12
    h = 34 + len(items) * row + 26
    W = label_w + bar_w + note_w
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria or title or "")}">']
    if title:
        s.append(f'<text class="ft" x="0" y="15">{esc(title)}</text>')
    s.append(f'<g transform="translate({label_w},30)">')
    for t in range(ticks + 1):
        x = bar_w * t / ticks
        val = vmax * t / ticks
        s.append(f'<line class="grid" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{len(items) * row}"/>')
        lab = (f"{val:.0f}" if vmax >= 10 else fmt.format(val)) + (unit if t == ticks else "")
        s.append(f'<text class="fa" x="{x:.1f}" y="{len(items) * row + 15}" text-anchor="middle">{lab}</text>')
    for n, it in enumerate(items):
        y = n * row
        s.append(f'<text class="fl" x="-10" y="{y + 19}" text-anchor="end">{esc(it["label"])}</text>')
        if it["v"] is None:
            s.append(f'<text class="fa" x="6" y="{y + 19}">{esc(it.get("note") or "—")}</text>')
            continue
        w = max(2.0, min(bar_w, it["v"] / vmax * bar_w))
        s.append(f'<rect x="0" y="{y + 6}" width="{w:.1f}" height="{row - 12}" rx="3" fill="{it["c"]}"><title>{esc(it.get("tip") or it["label"])}</title></rect>')
        lab = fmt.format(it["v"]) + (" · " + it["note"] if it.get("note") else "")
        s.append(f'<text class="fn" x="{w + 8:.1f}" y="{y + 19}">{esc(lab)}</text>')
    s.append("</g></svg>")
    return "".join(s)


def heat_class(v, vmax):
    t = v / vmax
    return "h1" if t < .12 else "h2" if t < .25 else "h3" if t < .45 else "h4" if t < .7 else "h5"


def modes_heat(compact=False):
    vmax = max(c["median"] for c in CELLS.values() if c.get("median"))
    rows = ['<table class="heat sortable"><thead><tr><th>model</th>' + "".join(f"<th>{ARM_LABEL[a]}</th>" for a in ARMS) + "</tr></thead><tbody>"]
    for key, name, route in MODELS:
        rows.append(f"<tr><td><b>{esc(name)}</b></td>")
        best = min((CELLS[(key, a)]["median"] for a in ARMS if (key, a) in CELLS), default=None)
        for a in ARMS:
            c = CELLS.get((key, a))
            if not c:
                rows.append('<td class="h h0">—</td>')
                continue
            hc = heat_class(c["median"], vmax)
            star = " ★" if c["median"] == best else ""
            sub = "" if compact else f'<span>{c["in"] // 1000}K in · {c["out"]} out</span>'
            rows.append(f'<td class="h {hc}" data-v="{c["median"]}">{c["median"]:.1f}{star}{sub}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def tbl(headers, rows, cls="sortable", num_cols=()):
    h = "".join(f'<th{" class=n" if i in num_cols else ""}>{esc(x)}</th>' for i, x in enumerate(headers))
    body = []
    for r in rows:
        tds = []
        for i, x in enumerate(r):
            x = "—" if x is None or x == "" else x
            tds.append(f'<td{" class=n" if i in num_cols else ""}>{x if str(x).startswith("<") else esc(str(x))}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def chip(txt):
    try:
        a, b = str(txt).split("/")[:2]
        a, b = int(a), int(b.split()[0])
        c = "g" if a == b else ("w" if a > 0 else "b")
    except Exception:
        c = ""
    return f'<span class="chip {c}">{esc(str(txt))}</span>'


# ------------------------------------------------------------------ derived
def gate_name(g):
    m = g["model"]
    nice = {"anthropic/claude-fable-5.1": "Claude Fable 5.1", "anthropic/claude-opus-5": "Claude Opus 5", "anthropic/claude-fable-5.1:batch": "Claude Fable 5.1 :batch",
            "inception/mercury-2.5": "Mercury 2.5", "x-ai/grok-4.6": "Grok 4.6", "openai/gpt-6-astra": "GPT-6 Astra"}.get(m, m)
    return nice + (f" · {g['effort']}" if g["effort"] else "")


LADDER = [dict(label=V41 + " (13 Sep)", v=14.8, c="var(--mint)", note="reference", tip="same three exercises, self-hosted B200"),
          dict(label=QFN + " (13 Sep)", v=17.0, c="var(--mint)", note="reference", tip="same three exercises, self-hosted B200")]
for g in sorted(GATE, key=lambda g: (g["v"]["coding_median_wall_s"] or 9e9)):
    v = g["v"]
    if v["correctness"].startswith("0/"):
        continue
    col = "var(--coral)" if g["route"] == "OpenRouter" else "var(--sky)"
    LADDER.append(dict(label=gate_name(g), v=v["coding_median_wall_s"], c=col, note=v["correctness"] + (" · at the cap" if v["coding_median_wall_s"] > 240 else ""),
                       tip=f"{gate_name(g)}: {v['coding_median_wall_s']} s · {v['decode_tps']} tok/s · {v['correctness']}"))

MEMRATIO = []
for key, name, route in MODELS:
    b, m = CELLS.get((key, "baseline")), CELLS.get((key, "mem"))
    if b and m:
        r = m["median"] / b["median"]
        MEMRATIO.append(dict(label=name, v=r, c="var(--mint)" if r <= 1.05 else ("var(--butter)" if r <= 1.6 else "var(--coral)"),
                             note=f"{m['median']:.1f} vs {b['median']:.1f} s", tip=name))

COST = []
for key, name, route in MODELS:
    c = CELLS.get((key, "baseline"))
    if c:
        cc = cell_cost(key, c)
        COST.append(dict(label=name, v=cc / c["n"], c={"local": "var(--mint)", "openrouter": "var(--coral)", "gateway": "var(--sky)"}[route],
                         note=("card share of the $28/h box" if route == "local" else f"{c['in'] // 1000}K in · {c['out']} out"), tip=name))
COST.sort(key=lambda x: x["v"])

# ------------------------------------------------------------------ shared css/js
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">'
SORT_JS = """
<script>
(function(){
  function val(td){ if(!td) return null; var t=(td.getAttribute('data-v')||td.textContent||'').trim();
    if(t===''||t==='-'||t==='—') return null;
    var f=t.match(/^(\\d+)\\/(\\d+)/); if(f) return parseInt(f[1],10)/Math.max(1,parseInt(f[2],10));
    var m=t.replace(/[$,%×\\s★]/g,'').match(/^-?\\d+(\\.\\d+)?/); if(m && /^[\\d$,.%×\\s★-]/.test(t)) return parseFloat(m[0]);
    return t.toLowerCase(); }
  document.querySelectorAll('table.sortable').forEach(function(tbl){
    var ths=tbl.querySelectorAll('thead th');
    ths.forEach(function(th,idx){ th.setAttribute('tabindex','0');
      function go(){ var dir=th.getAttribute('aria-sort')==='ascending'?'descending':'ascending';
        ths.forEach(function(o){o.removeAttribute('aria-sort')}); th.setAttribute('aria-sort',dir);
        var body=tbl.tBodies[0]; var rows=Array.prototype.slice.call(body.rows);
        rows.sort(function(a,b){ var va=val(a.cells[idx]), vb=val(b.cells[idx]);
          if(va===null&&vb===null) return 0; if(va===null) return 1; if(vb===null) return -1;
          if(typeof va==='number'&&typeof vb==='number') return dir==='ascending'?va-vb:vb-va;
          return dir==='ascending'?String(va).localeCompare(String(vb)):String(vb).localeCompare(String(va)); });
        rows.forEach(function(r){body.appendChild(r)}); }
      th.addEventListener('click',go); th.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}}); });
  });
})();
</script>
"""
