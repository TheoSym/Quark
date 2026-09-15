#!/usr/bin/env python
"""Run a subset of the Aider polyglot benchmark (Exercism exercises) through
coding agents against one OpenAI-compatible model endpoint, and score with
the exercise's own test file.

Agents:
  jcode  -- this repo's agent: `jcode run --json --provider-profile P --model M`
  codex  -- OpenAI Codex CLI: `codex exec` with CODEX_HOME pointing at a config
            whose model_provider is the same endpoint (wire_api = "responses").

Scoring is objective: after the agent's single turn, `python -m pytest -q` on
the exercise's *_test.py must exit 0. The prompt follows aider's benchmark
wording (instructions + "make the tests pass", stdlib only).

Example:
  python polyglot_bench.py --bench ../polyglot-benchmark --lang python \
      --exercises bowling wordy --agents jcode codex \
      --jcode ../../target/selfdev/jcode.exe --jcode-profile qwen38 \
      --model Qwen3.8-Flash-Next-NVFP4 --codex-home ./codex-home --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROMPT_TEMPLATE = """{instructions}

####

Use the above instructions to modify the supplied files: {files}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
{deps_line}
The unit tests are in {tests}; make them pass. Do not modify the test file.
"""

DEPS_STDLIB = "Only use standard {lang} libraries, don't suggest installing any packages."
DEPS_OPEN = ("You may pip install third-party {lang} packages or vendor MIT / Apache-2.0 licensed "
             "code into this directory if it gets the job done with less code; the tests run with "
             "the same python you have here.")

TEST_CMD = {
    "python": ["python", "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
}


@dataclass
class Result:
    agent: str
    exercise: str
    solved: bool
    wall_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None
    test_tail: str = ""
    agent_tail: str = ""


def load_exercise(bench: Path, lang: str, name: str) -> dict:
    d = bench / lang / "exercises" / "practice" / name
    if not d.is_dir():
        raise SystemExit(f"no such exercise: {d}")
    docs = d / ".docs"
    instr = ""
    for f in ("introduction.md", "instructions.md", "instructions.append.md"):
        p = docs / f
        if p.exists():
            instr += p.read_text(encoding="utf-8") + "\n\n"
    tests = sorted(p.name for p in d.glob("*_test.py"))
    stubs = sorted(p.name for p in d.glob("*.py") if not p.name.endswith("_test.py"))
    return {"dir": d, "instructions": instr.strip(), "tests": tests, "stubs": stubs}


def make_workdir(ex: dict, root: Path, agent: str, overlay: Path | None = None,
                 mcp: Path | None = None) -> Path:
    wd = Path(tempfile.mkdtemp(prefix=f"{agent}-{ex['dir'].name}-", dir=root))
    for p in ex["dir"].iterdir():
        if p.name == ".meta":  # holds the reference solution; never ship it
            continue
        if p.is_dir():
            shutil.copytree(p, wd / p.name)
        else:
            shutil.copy2(p, wd / p.name)
    if overlay or mcp:
        (wd / ".jcode").mkdir()
    if overlay:
        # jcode appends ./.jcode/prompt-overlay.md to its system prompt; this is
        # the caller-boundary seam for a qgi-2/prompts/coding-modes policy.
        shutil.copy2(overlay, wd / ".jcode" / "prompt-overlay.md")
    if mcp:
        # Project-scoped MCP servers (e.g. Exa for the exa-reuse policy); jcode
        # expands ${VAR} from the environment, so keys never land in the file.
        shutil.copy2(mcp, wd / ".jcode" / "mcp.json")
    subprocess.run(["git", "init", "-q", "."], cwd=wd, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "add", "-A"], cwd=wd, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-c", "user.email=b@b", "-c", "user.name=bench",
                    "commit", "-q", "-m", "start"], cwd=wd, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wd


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except OSError:
            os.kill(pid, 9)


def run_jcode(args, prompt: str, wd: Path, timeout: int) -> tuple[dict, str, str | None]:
    # Through the QGI-2 memory proxy, `model@key` selects a per-run memory
    # session (the proxy strips the suffix before forwarding).
    model = f"{args.model}@{wd.name}" if args.session_per_task else args.model
    cmd = [args.jcode, "run", "--json", "--provider-profile", args.jcode_profile,
           "--model", model, prompt]
    p = subprocess.Popen(cmd, cwd=wd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace",
                         stdin=subprocess.DEVNULL)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole tree: a test run the agent launched (e.g. an infinite
        # loop in its own solution) inherits our pipes and would otherwise
        # keep communicate() blocked forever after jcode itself is gone.
        kill_tree(p.pid)
        partial = ""
        try:
            out_p, err_p = p.communicate(timeout=30)
            partial = ((out_p or "") + (err_p or ""))[-600:]
        except subprocess.TimeoutExpired:
            pass
        # Keep what the agent had emitted: a task that passes its tests and
        # still hits the cap is the agent continuing after the fix, and the
        # tail is the only evidence of what it was doing.
        return {}, partial, f"timeout after {timeout}s"
    p.stdout, p.stderr = stdout, stderr
    out = p.stdout
    usage: dict = {}
    m = re.search(r"\{.*\}\s*$", out, re.S)
    if m:
        try:
            doc = json.loads(m.group(0))
            usage = doc.get("usage") or {}
        except json.JSONDecodeError:
            pass
    err = None if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}"
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_tokens": usage.get("cache_read_input_tokens"),
    }, (out + p.stderr)[-600:], err


def run_codex(args, prompt: str, wd: Path, timeout: int) -> tuple[dict, str, str | None]:
    env = dict(os.environ)
    env["CODEX_HOME"] = str(Path(args.codex_home).resolve())
    # npm installs `codex` as a shim; on Windows CreateProcess needs the .cmd.
    codex = (shutil.which(args.codex + ".cmd") if os.name == "nt" else None) \
        or shutil.which(args.codex) or args.codex
    cmd = [codex, "exec", "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox", "-C", str(wd), "-"]
    try:
        p = subprocess.run(cmd, cwd=wd, input=prompt, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {}, "", f"timeout after {timeout}s"
    out = p.stdout + "\n" + p.stderr
    usage: dict = {}
    m = re.search(r"tokens used\s*\n\s*([\d,]+)", out)
    if m:
        usage["input_tokens"] = int(m.group(1).replace(",", ""))  # codex reports one total
    err = None if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}"
    return {"input_tokens": usage.get("input_tokens"), "output_tokens": None,
            "cached_tokens": None}, out[-600:], err


def run_msa(args, prompt: str, wd: Path, timeout: int) -> tuple[dict, str, str | None]:
    """mini-swe-agent (SWE-agent/mini-swe-agent, or a fork): `mini -t TASK -y`
    with the litellm model pointed at the same OpenAI-compatible endpoint.
    Actions run under Git Bash via msa_bash_env (its prompts assume bash)."""
    env = dict(os.environ)
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["PYTHONUTF8"] = "1"                            # rich banner/emoji vs cp1252 console
    env["PYTHONIOENCODING"] = "utf-8"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"       # local models have no price entry
    env["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(args.msa_home).resolve()) if args.msa_home else env.get("MSWEA_GLOBAL_CONFIG_DIR", "")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OPENAI_API_KEY", "none")
    model = f"openai/{args.model}@{wd.name}" if args.session_per_task else f"openai/{args.model}"
    traj = wd / ".msa.traj.json"
    task_file = wd / ".msa.task.txt"
    task_file.write_text(prompt, encoding="utf-8")
    cmd = [args.msa_python, str(Path(__file__).resolve().parent / "msa_run.py"),
           "--model", model, "--api-base", args.msa_api_base,
           "--task-file", str(task_file), "--cwd", str(wd), "--out", str(traj),
           "--step-limit", str(args.msa_step_limit)]
    if args.msa_api_key or os.environ.get("MSA_API_KEY"):
        cmd += ["--api-key", args.msa_api_key or os.environ["MSA_API_KEY"]]
    if args.msa_config:
        cmd += ["--config", args.msa_config]
    if args.msa_extra_body:
        cmd += ["--extra-body", args.msa_extra_body]
    try:
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {}, "", f"timeout after {timeout}s"
    out = p.stdout + "\n" + p.stderr
    usage: dict = {"input_tokens": None, "output_tokens": None, "cached_tokens": None}
    m = re.search(r"\{[^\n]*\"n_calls\"[^\n]*\}\s*$", p.stdout.strip(), re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            usage = {k: d.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens")}
            out = f"n_calls={d.get('n_calls')} exit={d.get('exit_status')}\n" + out
        except json.JSONDecodeError:
            pass
    err = None if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}"
    return usage, out[-600:], err


def run_dsh(args, prompt: str, wd: Path, timeout: int) -> tuple[dict, str, str | None]:
    """DeepSeek Harness (deepseek-ai/deepseek-harness) through its Python SDK's
    `sdk-minimal` profile: one task, isolated dsh home, DEEPSEEK_BASE_URL at
    the endpoint. The provider is `deepseek-official`, so the wire format is
    DeepSeek's own (reasoning_content, reasoning_effort literals)."""
    env = dict(os.environ)
    env["DEEPSEEK_BASE_URL"] = args.dsh_api_base
    env["DEEPSEEK_API_KEY"] = args.dsh_api_key or env.get("DEEPSEEK_API_KEY") or "none"
    env["PYTHONUTF8"] = "1"
    home = Path(args.dsh_home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    task_file = wd / ".dsh.task.txt"
    task_file.write_text(prompt, encoding="utf-8")
    driver = Path(__file__).resolve().parent / "dsh_run.py"
    cmd = [args.dsh_python, str(driver), "--workspace", str(wd), "--dsh-home", str(home),
           "--session-id", wd.name, "--model", args.model, "--task-file", str(task_file)]
    try:
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {}, "", f"timeout after {timeout}s"
    out = p.stdout + "\n" + p.stderr
    usage: dict = {"input_tokens": None, "output_tokens": None, "cached_tokens": None}
    m = re.search(r"\{[^\n]*\"dsh_usage\"[^\n]*\}\s*$", p.stdout.strip(), re.S)
    if m:
        try:
            d = json.loads(m.group(0))["dsh_usage"]
            usage = {k: d.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens")}
        except (json.JSONDecodeError, KeyError):
            pass
    err = None if p.returncode == 0 else f"exit {p.returncode}: {p.stderr[-300:]}"
    return usage, out[-600:], err


def verify(ex: dict, wd: Path, lang: str) -> tuple[bool, str]:
    # Reset the test file in case the agent edited it.
    for t in ex["tests"]:
        shutil.copy2(ex["dir"] / t, wd / t)
    try:
        p = subprocess.run(TEST_CMD[lang] + ex["tests"], cwd=wd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    return p.returncode == 0, (p.stdout + p.stderr)[-400:]


def run_one(args, agent: str, name: str, root: Path) -> Result:
    ex = load_exercise(Path(args.bench), args.lang, name)
    overlay = Path(args.jcode_overlay) if (agent == "jcode" and args.jcode_overlay) else None
    mcp = Path(args.jcode_mcp) if (agent == "jcode" and args.jcode_mcp) else None
    wd = make_workdir(ex, root, agent, overlay, mcp)
    deps = (DEPS_OPEN if args.allow_packages else DEPS_STDLIB).format(lang=args.lang)
    prompt = PROMPT_TEMPLATE.format(instructions=ex["instructions"],
                                    files=", ".join(ex["stubs"]), lang=args.lang,
                                    tests=", ".join(ex["tests"]), deps_line=deps)
    t0 = time.time()
    runner = {"jcode": run_jcode, "codex": run_codex, "msa": run_msa, "dsh": run_dsh}[agent]
    usage, tail, err = runner(args, prompt, wd, args.timeout)
    wall = time.time() - t0
    solved, test_tail = verify(ex, wd, args.lang)
    r = Result(agent=args.label or agent, exercise=name, solved=solved, wall_s=round(wall, 1),
               error=err, test_tail=test_tail, agent_tail=tail, **usage)
    print(f"[{agent:5}] {name:28} {'PASS' if solved else 'FAIL':4} {wall:6.1f}s"
          f"{'  ' + err if err else ''}", flush=True)
    if not args.keep:
        shutil.rmtree(wd, ignore_errors=True)
    return r


def summarize(results: list[Result]) -> dict:
    by: dict[str, list[Result]] = {}
    for r in results:
        by.setdefault(r.agent, []).append(r)
    out = {}
    for agent, rs in by.items():
        out[agent] = {
            "n": len(rs),
            "solved": sum(r.solved for r in rs),
            "solve_rate": round(sum(r.solved for r in rs) / len(rs), 3),
            "median_wall_s": round(statistics.median(r.wall_s for r in rs), 1),
            "total_wall_s": round(sum(r.wall_s for r in rs), 1),
            "errors": sum(1 for r in rs if r.error),
            "input_tokens": sum(r.input_tokens or 0 for r in rs),
            "output_tokens": sum(r.output_tokens or 0 for r in rs),
            "cached_tokens": sum(r.cached_tokens or 0 for r in rs),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="path to Aider-AI/polyglot-benchmark checkout")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--allow-packages", action="store_true",
                    help="drop aider's stdlib-only line; tell the agent it may pip install or vendor "
                         "MIT/Apache-2.0 code (lets a search-and-reuse policy actually fire)")
    ap.add_argument("--exercises", nargs="*", help="exercise names (default: all)")
    ap.add_argument("--agents", nargs="+", default=["jcode", "codex"], choices=["jcode", "codex", "msa", "dsh"])
    ap.add_argument("--dsh-python", default="python", help="python of the venv with deepseek-harness-sdk")
    ap.add_argument("--dsh-api-base", default=None, help="DEEPSEEK_BASE_URL for the dsh agent (.../v1)")
    ap.add_argument("--dsh-api-key", default=None)
    ap.add_argument("--dsh-home", default=None, help="isolated DSH_HOME directory for the run")
    ap.add_argument("--msa-python", default="python", help="python of the venv where mini-swe-agent is installed")
    ap.add_argument("--msa-api-base", default=None, help="OpenAI-compatible base URL (.../v1) for the msa agent")
    ap.add_argument("--msa-config", default=None, help="mini-swe-agent config yaml (its builtin mini.yaml by default)")
    ap.add_argument("--msa-home", default=None, help="MSWEA_GLOBAL_CONFIG_DIR for the run")
    ap.add_argument("--msa-api-key", default=None, help="bearer for the endpoint (or env MSA_API_KEY); local engines need none")
    ap.add_argument("--msa-step-limit", type=int, default=40)
    ap.add_argument("--msa-extra-body", default=None, help='JSON merged into every request, e.g. {"chat_template_kwargs":{"enable_thinking":false}}')
    ap.add_argument("--model", required=True)
    ap.add_argument("--jcode", default="jcode")
    ap.add_argument("--jcode-profile", default="qwen38")
    ap.add_argument("--jcode-overlay", default=None,
                    help="jcode: system-prompt policy file copied to <workdir>/.jcode/prompt-overlay.md "
                         "(e.g. qgi-2/prompts/coding-modes/safe-efficient-coding.system.md)")
    ap.add_argument("--jcode-mcp", default=None,
                    help="jcode: MCP config copied to <workdir>/.jcode/mcp.json "
                         "(e.g. qgi-2/prompts/coding-modes/exa.mcp.json with EXA_API_KEY in the env)")
    ap.add_argument("--codex", default="codex")
    ap.add_argument("--codex-home", default=None, help="CODEX_HOME with a config.toml pointing at the endpoint")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--session-per-task", action="store_true",
                    help="jcode: append @<workdir> to the model so a QGI-2 proxy keys memory per run")
    ap.add_argument("--label", default=None, help="agent label override in results (e.g. jcode+qgi2)")
    ap.add_argument("--parallel-agents", action="store_true",
                    help="run each agent's sequence concurrently (same load on both arms)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = Path(args.bench)
    names = args.exercises or sorted(
        p.name for p in (bench / args.lang / "exercises" / "practice").iterdir() if p.is_dir())
    if "codex" in args.agents and not args.codex_home:
        raise SystemExit("--codex-home is required for the codex agent")

    root = Path(tempfile.mkdtemp(prefix="polyglot-bench-"))
    results: list[Result] = []
    lock = threading.Lock()

    def run_agent(agent: str) -> None:
        for name in names:
            r = run_one(args, agent, name, root)
            with lock:
                results.append(r)
                if args.out:
                    Path(args.out).write_text(json.dumps(
                        {"model": args.model, "lang": args.lang, "exercises": names,
                         "summary": summarize(results),
                         "results": [asdict(x) for x in results]}, indent=2))

    if args.parallel_agents and len(args.agents) > 1:
        threads = [threading.Thread(target=run_agent, args=(a,)) for a in args.agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for a in args.agents:
            run_agent(a)

    summary = summarize(results)
    print()
    print(f"{'agent':6} {'solved':>8} {'rate':>6} {'median s':>9} {'total s':>8} {'errors':>6} {'in tok':>10} {'out tok':>8} {'cached':>10}")
    for agent, s in summary.items():
        print(f"{agent:6} {s['solved']:>4}/{s['n']:<3} {s['solve_rate']:>6.0%} {s['median_wall_s']:>9} {s['total_wall_s']:>8} {s['errors']:>6} {s['input_tokens']:>10,} {s['output_tokens']:>8,} {s['cached_tokens']:>10,}")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
