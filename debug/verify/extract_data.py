#!/usr/bin/env python3
"""Extract embedding and attn_norm from mono27b_chat debug output.
Called by Makefile's generate-data target."""
import sys

debug_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mono_verify.debug.tsv'

with open(debug_path, 'r') as f:
    content = f.read()

saved = 0
for line in content.split('\n'):
    if line.startswith('embed\t0\t0\t'):
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue
        vals = [float(v) for v in parts[10].split(',')]
        with open('embed_full.txt', 'w') as f:
            for v in vals:
                f.write(f'{v:.10f}\n')
        print(f'Saved embed_full.txt ({len(vals)} vals)')
        saved += 1
    if '\tssm\t0\t0\t' in line and '\tattn_norm\t5120' in line:
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue
        vals = [float(v) for v in parts[10].split(',')]
        with open('attn_norm_full.txt', 'w') as f:
            for v in vals:
                f.write(f'{v:.10f}\n')
        print(f'Saved attn_norm_full.txt ({len(vals)} vals)')
        saved += 1

if saved < 2:
    print(f"Warning: only saved {saved}/2 data files. Debug output may be incomplete.")
    sys.exit(0 if saved > 0 else 1)
