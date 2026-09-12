#!/usr/bin/env python3
"""Regenerate models.lock.json from Hugging Face.

Run this only when a checkpoint is deliberately changed. The lock file is what
makes a measured cache-hit or acceptance number reproducible: it names the
exact revision and the sha256 of every weight file for each role.

Usage:  python3 pin-weights.py            # rewrite models.lock.json in place
        python3 pin-weights.py --check    # exit 1 if HF's current revision moved
"""
import datetime
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

LOCK = Path(__file__).with_name("models.lock.json")

# role -> repo. Change a repo here, then rerun; do not edit the lock by hand.
ROLES = {
    "planner": "deepseek-ai/DeepSeek-V4.1-Flash",
    "worker": "Qwen/Qwen3.8-27B-FP8",
    "worker_drafter": "incoai/Qwen3.8-27B-DFlash2",
}


def pin(api: HfApi, repo: str) -> dict:
    info = api.model_info(repo, files_metadata=True)
    files = list(info.siblings)
    return {
        "repo": repo,
        "revision": info.sha,
        "license": getattr(info.card_data, "license", None) if info.card_data else None,
        "files": len(files),
        "bytes": sum((s.size or 0) for s in files),
        "sha256": {s.rfilename: s.lfs.sha256 for s in files if s.lfs},
    }


def main() -> None:
    api = HfApi()
    fresh = {role: pin(api, repo) for role, repo in ROLES.items()}
    if "--check" in sys.argv:
        current = json.loads(LOCK.read_text(encoding="utf-8"))["roles"]
        moved = [r for r in ROLES if current.get(r, {}).get("revision") != fresh[r]["revision"]]
        for r in moved:
            print(f"{r}: pinned {current[r]['revision'][:12]} but HF main is now {fresh[r]['revision'][:12]}")
        sys.exit(1 if moved else 0)
    out = {
        "_comment": (
            "Pinned checkpoints for QGI-2. Weights never live in git; this file does. "
            "fetch-weights.sh downloads a role at `revision` and verifies every `sha256`. "
            "Regenerate with deploy/pin-weights.py after a deliberate model change, never as a side effect."
        ),
        "generated": datetime.date.today().isoformat(),
        "roles": fresh,
    }
    LOCK.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for role, r in fresh.items():
        print(f"{role:15s} {r['repo']:40s} {r['revision'][:12]}  {r['files']:3d} files  {r['bytes'] / 1e9:6.1f} GB")


if __name__ == "__main__":
    main()
