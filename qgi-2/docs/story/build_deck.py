from build_story import *  # noqa: F401,F403

CSS = """
:root{--bg:#0E1430;--panel:#18214D;--panel2:#1F2A5E;--ink:#F4F1EA;--ink2:#B8BEDA;--ink3:#8089B0;--rule:#2B3670;--grid:#26306A;
--coral:#FF6B4A;--butter:#FFD166;--mint:#4FD1A5;--lilac:#B9A7FF;--sky:#6FB3FF;
--h0:#18214D;--h1:#1D3E6B;--h2:#2A5C99;--h3:#3F82C9;--h4:#E0873F;--h5:#FF6B4A;
--disp:"Bricolage Grotesque","Arial Narrow",system-ui,sans-serif;--sans:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-snap-type:y mandatory;scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:clamp(15px,1.25vw,21px);line-height:1.45;-webkit-font-smoothing:antialiased}
.slide{min-height:100vh;scroll-snap-align:start;padding:6vh 6vw 8vh;display:grid;align-content:center;gap:2.4vh;position:relative;overflow:hidden}
.slide::after{content:attr(data-n) " / 10";position:absolute;right:6vw;bottom:3.2vh;font-family:var(--mono);font-size:.7em;color:var(--ink3);letter-spacing:.08em}
.eyebrow{font-family:var(--mono);font-size:.72em;letter-spacing:.14em;text-transform:uppercase;color:var(--butter);margin:0}
h1{font-family:var(--disp);font-weight:800;font-size:clamp(44px,6.4vw,112px);line-height:.95;letter-spacing:-.02em;margin:0;text-wrap:balance}
h2{font-family:var(--disp);font-weight:700;font-size:clamp(30px,3.7vw,64px);line-height:1.02;letter-spacing:-.015em;margin:0;text-wrap:balance;max-width:22ch}
h2 em{font-style:normal;color:var(--coral)}
.lede{font-size:1.25em;color:var(--ink2);max-width:52ch;margin:0}
p{margin:0;color:var(--ink2);max-width:60ch}p b,li b{color:var(--ink);font-weight:600}
.cols{display:grid;grid-template-columns:1fr 1.25fr;gap:4vw;align-items:center}
.cols.even{grid-template-columns:1fr 1fr}.cols.wide{grid-template-columns:.8fr 1.6fr}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:1.6vw}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:2.2vh 1.6vw}
.card h3{font-family:var(--disp);font-weight:700;font-size:1.25em;margin:0 0 .5em;color:var(--ink)}
.card.c{border-top:5px solid var(--coral)}.card.m{border-top:5px solid var(--mint)}.card.s{border-top:5px solid var(--sky)}.card.l{border-top:5px solid var(--lilac)}.card.b{border-top:5px solid var(--butter)}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:1.2vw;margin-top:2vh}
.stat{border-top:3px solid var(--rule);padding-top:1.2vh}
.stat b{display:block;font-family:var(--disp);font-weight:800;font-size:2.1em;line-height:1;color:var(--ink)}
.stat span{font-family:var(--mono);font-size:.62em;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3)}
.big{font-family:var(--disp);font-weight:800;font-size:clamp(60px,9vw,160px);line-height:.9;letter-spacing:-.03em}
.big small{display:block;font-family:var(--mono);font-weight:400;font-size:.12em;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);margin-top:1.4em;white-space:nowrap}
ul{margin:0;padding-left:1.1em;color:var(--ink2)}li{margin:.45em 0}
figure{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:2vh 1.4vw 1.2vh}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:.72em;color:var(--ink3);margin-top:.8vh}
.ft{font-family:var(--sans);font-size:17px;font-weight:600;fill:var(--ink)}.fl{font-family:var(--sans);font-size:15.5px;fill:var(--ink2)}
.fn{font-family:var(--mono);font-size:14.5px;font-weight:500;fill:var(--ink)}.fa{font-family:var(--mono);font-size:12.5px;fill:var(--ink3)}.grid{stroke:var(--grid)}
.legend{display:flex;gap:1.4vw;flex-wrap:wrap;font-family:var(--mono);font-size:.62em;color:var(--ink2)}
.legend i{display:inline-block;width:.9em;height:.9em;border-radius:3px;vertical-align:-1px;margin-right:.5em}
table{border-collapse:collapse;width:100%;font-size:.78em}
th,td{padding:.55em .6em;border-bottom:1px solid var(--rule);text-align:left}
th{font-family:var(--mono);font-size:.78em;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--ink3)}
.heat td.h{font-family:var(--mono);text-align:center;font-weight:500;color:#fff}
.heat td.h span{display:none}
.h0{background:var(--h0)}.h1{background:var(--h1)}.h2{background:var(--h2)}.h3{background:var(--h3)}.h4{background:var(--h4)}.h5{background:var(--h5)}
.gpus{display:grid;grid-template-columns:repeat(4,1fr);gap:1vw;height:34vh;align-items:end}
.gpu{display:flex;flex-direction:column-reverse;height:100%;background:var(--panel2);border-radius:10px;overflow:hidden;border:1px solid var(--rule)}
.gpu div{display:grid;place-items:center;font-family:var(--mono);font-size:.62em;color:#0E1430;font-weight:500;text-align:center;padding:2px}
.gpu-l{font-family:var(--mono);font-size:.62em;color:var(--ink3);text-align:center;margin-top:.6vh}
.blooper{display:grid;grid-template-columns:repeat(2,1fr);gap:1.2vw}
.blooper div{background:var(--panel);border-left:4px solid var(--butter);border-radius:0 10px 10px 0;padding:1.2vh 1vw;font-size:.82em;color:var(--ink2)}
.blooper b{display:block;font-family:var(--mono);font-size:.85em;color:var(--butter);margin-bottom:.2em}
a{color:var(--butter);text-underline-offset:4px}a:focus-visible{outline:2px solid var(--butter);outline-offset:3px}
.hint{position:fixed;left:6vw;bottom:3.2vh;font-family:var(--mono);font-size:11px;color:var(--ink3);letter-spacing:.06em}
@media (max-width:900px){.cols,.cols.even,.cols.wide,.three{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.blooper{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""

JS = """
<script>
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide'));
  var only=new URLSearchParams(location.search).get('only');
  if(only){ slides.forEach(function(s,i){ if(String(i+1)!==only) s.hidden=true; }); document.querySelector('.hint').hidden=true; return; }
  function cur(){ var y=window.scrollY+window.innerHeight/2; for(var i=0;i<slides.length;i++){ var r=slides[i]; if(y>=r.offsetTop && y<r.offsetTop+r.offsetHeight) return i; } return 0; }
  document.addEventListener('keydown',function(e){
    var k=e.key, i=cur();
    if(k==='ArrowRight'||k==='ArrowDown'||k==='PageDown'||k===' '){ e.preventDefault(); slides[Math.min(slides.length-1,i+1)].scrollIntoView(); }
    if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'){ e.preventDefault(); slides[Math.max(0,i-1)].scrollIntoView(); }
    if(k==='Home'){ slides[0].scrollIntoView(); } if(k==='End'){ slides[slides.length-1].scrollIntoView(); }
  });
})();
</script>
"""

ab_items = []
for label, d, q, td, tq, cache, wd, wq in AB[1:]:
    short = label.split(" · ")[0]
    ab_items.append(dict(label=f"pass {short} · jcode direct", v=wd, c="var(--sky)", note=d))
    ab_items.append(dict(label=f"pass {short} · through QGI-2", v=wq, c="var(--coral)" if wq > 20 else "var(--lilac)", note=q))
mem_items = [dict(label=f"{'DS V4.1' if m == V41 else 'Flash-Next'} · {arm}", v=float(rec), c=("var(--ink3)" if arm.startswith("whole") else "var(--coral)" if arm.startswith("12") else "var(--lilac)"),
                  note=f"{tok // 1000}K prompt tokens") for m, arm, rec, tok, lat in MEM]
for it in mem_items:
    if it["v"] == 0:
        it["v"] = 0.15
acc_items = [dict(label=l.replace(QFN, "Flash-Next").replace(V41, "DS V4.1"), v=hi, c="var(--coral)" if hi < 1.8 else "var(--mint)", note=(f"range {lo}–{hi}" if lo != hi else "")) for l, lo, hi, n in ACCEPT]

S = []
S.append(f"""<section class="slide" data-n="1"><p class="eyebrow">QGI-2 · Quark · 11–17 September 2026</p>
<h1>Four Days on<br>Rented Silicon</h1>
<p class="lede">One harness, two self-hosted heroes, fourteen rivals, and a credit card that tapped out with $1.85 left.</p>
<div class="stats"><div class="stat"><b>2</b><span>GPU boxes · one lost</span></div><div class="stat"><b>24</b><span>models touched</span></div><div class="stat"><b>402</b><span>harness tests</span></div>
<div class="stat"><b>16×15</b><span>gate verdicts</span></div><div class="stat"><b>48</b><span>matrix cells, all 3/3</span></div><div class="stat"><b>$123</b><span>OpenRouter, two days</span></div></div></section>""")

S.append(f"""<section class="slide" data-n="2"><p class="eyebrow">The plan</p><h2>A spec, a harness, and a CLI that <em>never touches jcode</em></h2>
<div class="three"><div class="card b"><h3>The spec</h3><p>Planner and worker. Six prompt segments, the first three byte-stable so the prefix cache hits. Typed facts the model proposes and rules commit. Every step a (model, speculation, sampling) triple; nothing defaults.</p></div>
<div class="card l"><h3>QGI-2</h3><p>Eleven crates beside jcode, <b>zero jcode edits</b>. Engine trait for vLLM and SGLang, HiCache and GDN-pool sizing, two edges, a scripted engine that makes the loop testable: <b>402 tests</b>. Datalog went in, then out: 15 rules, one recursive.</p></div>
<div class="card m"><h3>Quark</h3><p>jcode, branded and wired for the app factory: a stdlib CLI with its own home and hub profile, a runner that speaks AG-UI, a <b>9K-token</b> prompt instead of 33K, and one prompt policy that is opt-in only.</p></div></div>
<p style="max-width:none">The heroes of this story, as billed: <b>{V41}</b> and <b>{QFN}</b>.</p></section>""")

S.append(f"""<section class="slide" data-n="3"><p class="eyebrow">Act I · infrastructure</p><h2>The box that couldn't <em>count to three</em></h2>
<div class="cols even"><div><ul>
<li><b>8× RTX PRO 6000.</b> CUDA starts for any one or two GPUs, fails for any three. Host driver bug. Then a VM conversion destroyed the box.</li>
<li><b>4× B200, $28 an hour.</b> 510 GB planner pulled at 5 GB/s. First boot: <b>35 minutes</b> of kernel JIT, five after the caches warm.</li>
<li><b>TP4 was mostly air.</b> 556 GB of host RAM pinned for a cache tier that never hit. TP2 at 0.92 fits, DSpark 2.9. TP3 is impossible: a vocabulary of 129,280 does not divide by&nbsp;three.</li>
<li>Before shutdown: 488-file reproduction snapshot, 209 MB of JIT caches.</li></ul></div>
<div><div class="gpus">
<div class="gpu"><div style="height:91%;background:var(--mint)">DS V4.1<br>TP2 · 166 GB</div></div>
<div class="gpu"><div style="height:91%;background:var(--mint)">DS V4.1<br>TP2 · 166 GB</div></div>
<div class="gpu"><div style="height:89%;background:var(--sky)">Flash-Next<br>MTP · 163 GB</div></div>
<div class="gpu"><div style="height:30%;background:var(--lilac)">27B · 0.30</div><div style="height:55%;background:var(--butter)">Qwen3.6 · 0.55</div><div style="height:13%;background:var(--coral)">DGemma</div></div>
</div><div class="gpus" style="height:auto"><div class="gpu-l">card 0</div><div class="gpu-l">card 1</div><div class="gpu-l">card 2</div><div class="gpu-l">card 3 · shared</div></div></div></div></section>""")

S.append(f"""<section class="slide" data-n="4"><p class="eyebrow">Act II · the harness</p><h2>We built a planner. <em>jcode beat it</em> by 28 seconds.</h2>
<div class="cols"><div><p class="lede">Same model, same tasks, jcode direct versus jcode through QGI-2.</p><ul>
<li><b>Pass 1:</b> 1/4. Every failure ours: invented tool names, a JSON body where a stream was due, one session shared by everyone.</li>
<li><b>Pass 2:</b> 6/8 at 33.8 s. Plan → tool-args → extract is <b>three engine calls a round</b> where jcode spends one.</li>
<li><b>Pass 3:</b> stop planning, only remember. 8/8, 5.4 s, +4 % tokens.</li></ul></div>
<figure>{hbars(ab_items, title="Median seconds per task", label_w=260, bar_w=400, vmax=36, row=38, note_w=110)}<figcaption>Coral is the step-split harness; lilac is the memory-only proxy that replaced it.</figcaption></figure></div></section>""")

S.append(f"""<section class="slide" data-n="5"><p class="eyebrow">Act III · memory</p><h2>Eighty-eight turns later, it still knows the <em>Wi-Fi password</em></h2>
<div class="cols wide"><div><div class="big">0 → 20<small>facts recalled · 12-turn window</small></div>
<p style="margin-top:2vh">A hundred turns of small talk, twenty facts planted early. Keep only the last twelve turns and the bare model recalls <b>nothing</b>. Add the proxy: <b>20/20</b> and <b>18/20</b>, on <b>44 %</b> and <b>39 %</b> of the whole-transcript token bill.</p></div>
<figure>{hbars(mem_items, title="Recall out of 20 · prompt tokens over the run", label_w=300, bar_w=330, vmax=22, row=38, fmt="{:.0f}", note_w=230)}<figcaption>The first version the same morning scored 13 and 10. A spaCy ingester and incremental compaction closed the gap; no model changed.</figcaption></figure></div></section>""")

S.append(f"""<section class="slide" data-n="6"><p class="eyebrow">The locals</p><h2>Two self-hosted models that <em>held the line</em></h2>
<div class="cols" style="grid-template-columns:1.15fr 1fr"><div class="three" style="grid-template-columns:1fr 1fr"><div class="card m"><h3>{V41}</h3><ul>
<li><b>254 tok/s</b> single stream, <b>1,337</b> at eight</li><li>TTFT <b>0.18 s</b> on a 7.7K prompt</li><li>Documents: <b>58/58</b> checks in <b>9.6 min</b>; Gemini 12.3, DeepSeek V4 Pro via gateway 74</li><li><b>14.8 s</b> per exercise</li><li>Four speed levers tried, none beat baseline</li></ul></div>
<div class="card s"><h3>{QFN}</h3><ul><li>Same engine, same flags, different export: MTP accept <b>1.5 → 3.5</b></li><li>Polyglot <b>16/16</b> at 16.0 s; codex on the same card 33.3 s</li><li>157 tok/s with MTP, fp8 KV, 3 M-token pool</li><li>Likes the safe-efficient diet: 26 → <b>15.2 s</b></li></ul></div></div>
<figure>{hbars(acc_items, title="Speculation: accepted tokens per step", label_w=300, bar_w=300, vmax=7, row=44, note_w=170)}<figcaption>Spec floors: planner 1.8, worker 2.0. The spec's own table said DFlash2 cannot run greedy. It runs greedy.</figcaption></figure></div></section>""")

S.append(f"""<section class="slide" data-n="7"><p class="eyebrow">The gate · 15 verdicts in 15 minutes</p><h2>Everyone gets an A. <em>Speed</em> does the grading.</h2>
<div class="cols wide"><div><p>Three exercises, two documents, a two-turn task, eight facts from a 40-turn context. <b>Fourteen of sixteen</b> models scored 15/15.</p>
<ul><li>Opus at max: 13/15, both misses a <b>content filter</b></li><li>Fable :batch: batch-API-only, 404</li><li>QGI-Flash 3.8 Next: answered nothing, 0/15</li><li><b>Mercury:</b> 342 tok/s, one cent a gate</li></ul>
<div class="legend" style="margin-top:2vh"><span><i style="background:var(--mint)"></i>self-hosted</span><span><i style="background:var(--coral)"></i>direct on OpenRouter</span><span><i style="background:var(--sky)"></i>gateway</span></div></div>
<figure>{hbars(LADDER, title="Median seconds per coding task", label_w=330, bar_w=380, vmax=240, row=27, fmt="{:.1f}", note_w=230)}<figcaption>DeepSeek V4 Pro via the gateway sits at the 600 s cap (clipped): it passes its tests, then keeps going.</figcaption></figure></div></section>""")

S.append(f"""<section class="slide" data-n="8"><p class="eyebrow">The matrix · 8 models × 6 arms</p><h2>Prompt diets help some. The <em>memory tax</em> hits others.</h2>
<div class="cols wide"><div><ul><li><b>48 cells, all 3/3.</b> Correctness never moved; only seconds and tokens did.</li><li><b>safe-efficient</b> speeds Gemini, Opus, GLM and both locals; slows Fable and QGI 3.8 Flash.</li>
<li><b>Memory proxy:</b> free on Fable (23.5 vs 24.3 s), a gift to QGI-V4 Pro, a <b>4.8× tax</b> on GLM 5.3.</li><li><b>Exa:</b> connected on 48 runs, called <b>zero</b> times.</li></ul></div>
<figure>{modes_heat(compact=True)}<figcaption>Median seconds per exercise. ★ marks each model's fastest arm; cool is fast, coral is slow. Local rows ran on the 13th and have no memory arms.</figcaption></figure></div></section>""")

S.append(f"""<section class="slide" data-n="9"><p class="eyebrow">The bill</p><h2>What the agent reports and <em>what the card is charged</em></h2>
<div class="cols even"><figure>{hbars([dict(i, note="") for i in COST], title="USD per exercise, baseline arm", label_w=310, bar_w=300, row=40, fmt="${:.4f}", note_w=100)}<figcaption>Token-priced from each rate card; locals by their share of the $28/h box. Floors: reasoning tokens are not counted.</figcaption></figure>
<div><div class="three" style="grid-template-columns:1fr 1fr"><div class="card c"><h3>$46.99</h3><p>charged for six frontier gates. Priced from reported tokens: <b>$5.83</b>. Reasoning tokens never reach the agent's counters.</p></div>
<div class="card b"><h3>$1.85</h3><p>left when every call turned 402. The gateway's online models bill the <b>same OpenRouter account</b>.</p></div>
<div class="card m"><h3>1¢</h3><p>Mercury 2.5, whole gate, 15/15.</p></div><div class="card l"><h3>$58.92</h3><p>the 48-cell matrix after the top-up. Output tokens: under 1,000 per three exercises. You pay for context.</p></div></div></div></div></section>""")

DECK_BLOOPERS = [("HR-229", "Asked for ticket HR-2291, got \u201cHR-229\u201d: reasoning tokens ate the 120-token cap."), ("content_filter", "Opus at max returned two empty answers to a recall question. Its only misses."), ("Finish, then keep going", "Seven models passed their tests and kept working to the 600 s cap."), ("Nobody called Exa", "48 tool-arm runs, connected every time, zero calls.")]
bl = "".join(f"<div><b>{esc(t)}</b>{esc(x)}</div>" for t, x in DECK_BLOOPERS)
S.append(f"""<section class="slide" data-n="10"><p class="eyebrow">What we keep</p><h2>Remember more, plan less, <em>host your own</em></h2>
<div class="cols even"><div><ul><li><b>The memory layer stays.</b> Free on the models that matter, decisive on long sessions.</li><li><b>The step-split planner goes.</b> Native tool calling wins.</li>
<li><b>Self-hosting wins on speed:</b> 14.8 s against 24.3 for the best frontier model and 50+ for the gateway.</li><li><b>Effort is a lever, not a virtue.</b> Max and high bought minutes, not answers.</li>
<li><b>Prompt policy is per model.</b> So safe-efficient ships opt-in.</li><li><b>Quark ships:</b> CLI and runner in the app factory, 47/47 tests.</li></ul></div>
<div><p class="eyebrow" style="margin-bottom:1vh">Blooper reel</p><div class="blooper">{bl}</div></div></div></section>""")

page = "<title>Four Days on Rented Silicon · Deck</title>" + FONTS + "<style>" + CSS + "</style>" + "".join(S) + '<div class="hint">← → to move · 10 slides</div>' + JS
open(os.path.join(OUT, "quark-story-deck.html"), "w", encoding="utf-8").write(page)
print("deck written", len(page))
