#!/usr/bin/env python3
"""Master verification: runs local checks and compares against fixtures.

Usage: python3 run_all_checks.py --gguf PATH --debug PATH [--ref-intermediates PATH]
"""

from __future__ import annotations

import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

from trace_utils import extract_ref_tensor, get_debug_tensor, parse_debug_tsv

SCRIPT_DIR = Path(__file__).resolve().parent

# Parse args
GGUF = DEBUG = REF_INT = None
for i, a in enumerate(sys.argv):
    if a == '--gguf': GGUF = sys.argv[i+1]
    elif a == '--debug': DEBUG = sys.argv[i+1]
    elif a == '--ref-intermediates': REF_INT = sys.argv[i+1]

if not GGUF or not DEBUG:
    GGUF = GGUF or '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    DEBUG = DEBUG or '/tmp/mono_verify.debug.tsv'

if REF_INT and not os.path.isabs(REF_INT):
    REF_INT = str(SCRIPT_DIR / REF_INT)

results = {}

def run_py(script, *args):
    path = str(SCRIPT_DIR / script)
    cmd = [sys.executable, path, GGUF] + list(args)
    if DEBUG not in args:
        cmd.append(DEBUG)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc.stdout, proc.stderr, proc.returncode

def parse_match(stdout):
    return bool(re.search(r'MATCH:\s+YES', stdout)) or 'PASSED' in stdout

def extract_float(stdout, pattern):
    m = re.search(pattern, stdout)
    return float(m.group(1)) if m else None

# ================================================================
# 1. Q6_K format sanity
# ================================================================
out, err, rc = run_py('check_q6k_sanity.py')
results['q6k_sanity'] = {
    'status': 'PASS' if rc == 0 else 'FAIL',
    'detail': 'd values in valid range' if rc == 0 else err[:80],
}

# ================================================================
# 2. RMS norm
# ================================================================
out, err, rc = run_py('verify_rms_norm.py')
md = extract_float(out, r'max diff:?\s*([\d.]+(?:e[+-]?\d+)?)')
results['rms_norm'] = {
    'status': 'PASS' if parse_match(out) and rc == 0 else 'FAIL',
    'detail': f'max diff={md:.2e}' if md else 'N/A',
}

# ================================================================
# 3. Q5_K matvec
# ================================================================
out, err, rc = run_py('test_q5k_matvec.py')
diff = extract_float(out, r'Diff:\s+([\d.]+(?:e[+-]?\d+)?)')
cpu = extract_float(out, r'CPU dot:\s+([\d.-]+)')
gpu = extract_float(out, r'GPU val:\s+([\d.-]+)')
results['q5k'] = {
    'status': 'PASS' if parse_match(out) and rc == 0 else 'FAIL',
    'detail': f'diff={diff:.2e}' if diff else 'N/A',
    'cpu': cpu, 'gpu': gpu, 'diff': diff,
}

# ================================================================
# 4. Q6_K full matvec
# ================================================================
h2_path = os.path.join(SCRIPT_DIR, 'attn_norm_full.txt')
out, err, rc = run_py('verify_q6k_full.py', h2_path)
d = extract_float(out, r'diff:\s+([\d.]+(?:[eE][+-]?\d+)?)')
cq = extract_float(out, r'CPU dot:\s+([\d.-]+)')
gq = extract_float(out, r'GPU \[0\]:\s+([\d.-]+)')
results['q6k'] = {
    'status': 'PASS' if rc == 0 and d and d < 1e-3 else 'FAIL',
    'detail': f'diff={d:.2e}' if d else 'N/A',
    'cpu': cq, 'gpu': gq, 'diff': d,
}

# ================================================================
# 5. M-RoPE text rotation
# ================================================================
out, err, rc = run_py('verify_mrope.py')
worst = extract_float(out, r'worst_diff=([\d.]+(?:[eE][+-]?\d+)?)')
results['mrope'] = {
    'status': 'PASS' if rc == 0 else 'FAIL',
    'detail': f'worst_diff={worst:.2e}' if worst is not None else 'N/A',
}

# ================================================================
# 6. E2E logit comparison
# ================================================================
ref_bin = SCRIPT_DIR / 'ref_logits.bin'
our_bin = SCRIPT_DIR / 'our_logits.bin'

if ref_bin.exists() and our_bin.exists():
    with ref_bin.open('rb') as f:
        ref_l = struct.unpack(f'<248320f', f.read(248320*4))
    with our_bin.open('rb') as f:
        our_l = struct.unpack(f'<248320f', f.read(248320*4))
    n = len(ref_l)
    rm = sum(ref_l)/n; om = sum(our_l)/n
    corr = sum((ref_l[i]-rm)*(our_l[i]-om) for i in range(n))
    corr /= math.sqrt(sum((v-rm)**2 for v in ref_l) * sum((v-om)**2 for v in our_l))
    mse = sum((ref_l[i]-our_l[i])**2 for i in range(n))/n
    ref_top5 = sorted([(ref_l[i], i) for i in range(n)], reverse=True)[:5]
    our_top5 = sorted([(our_l[i], i) for i in range(n)], reverse=True)[:5]
    results['e2e'] = {
        'corr': corr, 'mse': mse,
        'ref_top5': [(f'{v:.2f}', i) for v,i in ref_top5],
        'our_top5': [(f'{v:.2f}', i) for v,i in our_top5],
    }
else:
    results['e2e'] = None

# ================================================================
# PRINT SUMMARY
# ================================================================
print()
print("=" * 78)
print("  MONO27B vs LLAMA.CPP — VERIFICATION SUMMARY")
print("=" * 78)

# Individual component checks
print(f"\n  {'Component Check':<35} {'Status':<10} {'Detail'}")
print("  " + "-" * 72)
for name, r in [('Q6_K format sanity', results['q6k_sanity']),
                ('RMS norm (5120 elem)', results['rms_norm']),
                ('Q5_K matvec (wqkv_gate)', results['q5k']),
                ('Q6_K matvec (wqkv)', results['q6k']),
                ('M-RoPE text rotation', results['mrope'])]:
    print(f"  {name:<35} {r['status']:<10} {r['detail']:<30}")

print()

# Numerical values table (matvec)
print(f"  {'Matvec':<30} {'CPU':>14} {'GPU':>14} {'Diff':>14}")
print("  " + "-" * 72)
for rkey, label in [('q5k', 'Q5_K wqkv_gate row 0'), ('q6k', 'Q6_K wqkv row 0')]:
    r = results.get(rkey, {})
    if r.get('cpu') and r.get('gpu'):
        print(f"  {label:<30} {r['cpu']:>14.8f} {r['gpu']:>14.8f} {r['diff']:>14.2e}")

print()

# E2E summary
if results['e2e']:
    e = results['e2e']
    print()
    print(f"  {'E2E Logits':<30} {'Ref':>20} {'Ours':>20}")
    print("  " + "-" * 72)
    print(f"  {'Top1 token':<30} {str(e['ref_top5'][0][1]):>20} {str(e['our_top5'][0][1]):>20}")
    print(f"  {'Top1 logit':<30} {e['ref_top5'][0][0]:>20} {e['our_top5'][0][0]:>20}")
    print(f"  {'Correlation':<30} {e['corr']:>20.6f}")
    print(f"  {'MSE':<30} {e['mse']:>20.4f}")

print()
print("=" * 78)
print("  Legend: PASS = matches ref within tolerance, DIFFERS = needs investigation")
print("=" * 78)
