#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mono27b_chat_bin> <model.gguf>" >&2
  exit 2
fi

BIN=$1
MODEL=$2
TRACE_OUT=/tmp/mono_decode_scaling_trace.txt
PROMPT="Count from 1 to 200, one per line, and do not stop early."

"$BIN" -m "$MODEL" -p "$PROMPT" --gen 80 --ctx 8192 --seed 944990222 --trace-gen --quiet > /tmp/mono_decode_scaling_stdout.txt 2>"$TRACE_OUT"

python3 - "$TRACE_OUT" <<'PY'
import sys
path = sys.argv[1]
vals = []
with open(path) as f:
    for line in f:
        if not line.startswith("[gen "):
            continue
        frag = line.split("decode_ms=", 1)[1].split()[0]
        vals.append(float(frag))
if len(vals) < 20:
    print(f"not enough trace rows: {len(vals)}", file=sys.stderr)
    sys.exit(1)
first = sum(vals[:10]) / 10.0
last = sum(vals[-10:]) / 10.0
delta = last - first
print(f"first_decode_ms={first:.3f}")
print(f"last_decode_ms={last:.3f}")
print(f"delta_ms={delta:.3f}")
if delta > 1.5:
    sys.exit(1)
PY
