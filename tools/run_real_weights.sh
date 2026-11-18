#!/usr/bin/env bash
# Compress every dump listed in a manifest and print ratios (no fake numbers).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: $0 data/gpt2/manifest.json}"
TC="${ROOT}/build/tc"
DIR="$(dirname "$MANIFEST")"

if [[ ! -x "$TC" ]]; then
  echo "build tc first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build" >&2
  exit 1
fi

python3 - "$MANIFEST" "$TC" "$DIR" <<'PY'
import json, subprocess, sys
from pathlib import Path

manifest_path, tc, directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
man = json.loads(Path(manifest_path).read_text())
print(f"model={man.get('model')} tensors={len(man['tensors'])}")
print(f"{'name':40} {'nbytes':>10} {'ratio':>8} {'upcast':>6}")
for t in man["tensors"]:
    raw = directory / t["file"]
    shape = ",".join(str(d) for d in t["shape"])
    out = directory / (t["file"] + ".tcf")
    subprocess.check_call(
        [tc, "compress", "-i", str(raw), "-o", str(out), "--dtype", "fp32", "--shape", shape],
        stdout=subprocess.DEVNULL,
    )
    stats = subprocess.check_output([tc, "stats", "-i", str(out)], text=True)
    ratio = None
    for line in stats.splitlines():
        if line.startswith("ratio:"):
            ratio = line.split()[1].rstrip("x")
    print(f"{t['name'][:40]:40} {t['nbytes']:10d} {ratio:>8} {str(t.get('upcast')):>6}")
PY
