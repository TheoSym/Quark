#!/usr/bin/env bash
# Fetch one role's checkpoint at the revision pinned in models.lock.json and
# verify every weight file's sha256 against the lock.
#
# Usage:  ./fetch-weights.sh <planner|worker|worker_drafter> [dest_root]
#
#   dest_root defaults to $MODELS_DIR, then /data/models. The checkpoint lands
#   in <dest_root>/<repo basename>, which is what the launch scripts expect in
#   MODEL_PATH.
#
# Env:
#   HF_TOKEN                 not needed -- all three repos are public
#   HF_HUB_ENABLE_HF_TRANSFER=1   several times faster on a fat pipe; pip install hf_transfer
#   VERIFY=0                 skip the digest pass (it re-reads every byte; ~510 GB for the planner)
#
# Sizes at the pinned revisions: planner ~510 GB, worker ~31 GB, drafter ~4 GB.
# Check free space first: the planner needs 600 GB with headroom for the
# download's temp files.
set -euo pipefail

ROLE="${1:?usage: fetch-weights.sh <planner|worker|worker_drafter> [dest_root]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/models.lock.json"
DEST_ROOT="${2:-${MODELS_DIR:-/data/models}}"

read -r REPO REV BYTES < <(python3 - "$LOCK" "$ROLE" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["roles"][sys.argv[2]]
print(r["repo"], r["revision"], r["bytes"])
EOF
)

DEST="$DEST_ROOT/$(basename "$REPO")"
NEED_GB=$(( BYTES / 1000000000 + BYTES / 5000000000 + 1 ))   # +20% headroom
FREE_GB=$(df -BG --output=avail "$DEST_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
mkdir -p "$DEST_ROOT"
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "refusing: $DEST_ROOT has ${FREE_GB} GB free, $ROLE needs ~${NEED_GB} GB" >&2
  exit 1
fi

echo "== $ROLE: $REPO @ ${REV:0:12} -> $DEST  (~$(( BYTES / 1000000000 )) GB)"

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --revision "$REV" --local-dir "$DEST"
else
  huggingface-cli download "$REPO" --revision "$REV" --local-dir "$DEST"
fi

if [ "${VERIFY:-1}" = "1" ]; then
  echo "== verifying sha256 against models.lock.json"
  python3 - "$LOCK" "$ROLE" "$DEST" <<'EOF'
import hashlib, json, sys
from pathlib import Path
lock, role, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
want = json.load(open(lock))["roles"][role]["sha256"]
bad = []
for i, (rel, digest) in enumerate(sorted(want.items()), 1):
    p = dest / rel
    if not p.exists():
        bad.append((rel, "missing")); continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        bad.append((rel, "mismatch"))
    print(f"  [{i}/{len(want)}] {rel} {'ok' if not bad or bad[-1][0] != rel else bad[-1][1]}", flush=True)
if bad:
    for rel, why in bad: print(f"FAIL {why}: {rel}", file=sys.stderr)
    sys.exit(1)
print(f"all {len(want)} digests match")
EOF
fi

# Record what was fetched beside the weights, so a launch can be tied to it.
printf '{"role":"%s","repo":"%s","revision":"%s","fetched":"%s"}\n' \
  "$ROLE" "$REPO" "$REV" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$DEST/.qgi2-pin.json"
echo "== done: $DEST"
