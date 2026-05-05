#!/usr/bin/env python3
"""
Comprehensive tensor comparison between mono27b debug output and llama.cpp reference.

Usage:
    python3 compare_all.py --our-debug /tmp/mono_verify.debug.tsv \
                           --ref /tmp/llama_ref.txt \
                           --ref-logits debug/verify/ref_logits.bin \
                           --our-logits debug/verify/our_logits.bin

Auto-discovers all comparable tensors and produces a detailed parity report.
"""

import argparse
import math
import os
import re
import struct
import sys
from pathlib import Path

# =============================================================================
# PARSERS
# =============================================================================

def parse_our_tsv(path):
    """Parse mono27b debug TSV into {(phase, step, label): [floats]}."""
    data = {}
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            phase = parts[0]
            try:
                step = int(parts[1])
            except ValueError:
                continue
            tok = parts[3]
            label = parts[4]
            try:
                n = int(parts[5])
            except ValueError:
                continue
            vals = [float(v) for v in parts[10].split(',') if v.strip() != '']
            if vals:
                data[(phase, step, label)] = vals
    return data


def parse_ref_stdout(path):
    """Parse llama-debug stdout; use LAST occurrence of each tensor."""
    data = {}
    with open(path) as f:
        content = f.read()

    # Split by tensor evaluation blocks
    blocks = re.split(r'common_debug_cb_eval:\s+', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Extract tensor name from first line
        m = re.match(r'(\S+)\s+=\s+\(f32\)', block)
        if not m:
            continue
        name = m.group(1)
        # Extract array content between first [ and matching ]
        arr_match = re.search(r'(\[.*?\])\s*\n\s*sum\s+=', block, re.DOTALL)
        if not arr_match:
            continue
        arr_str = arr_match.group(1)
        flat = arr_str.replace('[', ' ').replace(']', ' ').replace(',', ' ')
        try:
            vals = [float(x) for x in flat.split() if x.strip()]
        except ValueError:
            continue
        # Overwrite -> keeps LAST occurrence
        data[name] = vals
    return data


# =============================================================================
# LABEL MAPPING (bidirectional, auto-discoverable)
# =============================================================================

# Our (phase, label) -> reference name pattern (layer idx substituted)
OUR_TO_REF = {
    ('embed', 'h'): 'model.input_embed',
    ('ssm', 'attn_norm'): 'attn_norm-{layer}',
    ('ssm', 'z'): 'z-{layer}',
    ('ssm', 'conv_raw'): 'conv_output_raw-{layer}',
    ('ssm', 'conv'): 'conv_output_silu-{layer}',
    ('ssm', 'q_conv_predelta'): 'q_conv_predelta-{layer}',
    ('ssm', 'k_conv_predelta'): 'k_conv_predelta-{layer}',
    ('ssm', 'deltanet'): 'attn_output-{layer}',
    ('ssm', 'final_output'): 'final_output-{layer}',
    ('ssm', 'post_norm'): 'post_attention_norm-{layer}',
    ('ssm', 'ffn_gate'): 'ffn_gate-{layer}',
    ('ssm', 'ffn_up'): 'ffn_up-{layer}',
    ('ssm', 'ffn_down'): 'ffn_down-{layer}',
    ('ssm', 'layer_out'): 'l_out-{layer}',
    ('ssm', 'wqkv_gate'): None,  # no direct ref equivalent
    ('ssm', 'wqkv'): None,
    ('ssm', 'linear_attn_qkv_mixed'): None,
    ('ssm', 'beta'): None,
    ('ssm', 'dt'): None,
    ('ssm', 'gate'): None,
    ('ssm', 'rms_gated'): None,
    ('ssm', 'ssm_out'): None,
    ('ssm', 'post_ffn'): None,
    ('out', 'logits'): None,  # handled separately
    ('out', 'output_norm'): None,
    ('attn', 'rms'): 'attn_norm-{layer}',
    ('attn', 'attn_out'): 'attn_output-{layer}',
    ('attn', 'post_norm'): 'attn_post_norm-{layer}',
    ('attn', 'ffn_gate'): 'ffn_gate-{layer}',
    ('attn', 'ffn_up'): 'ffn_up-{layer}',
    ('attn', 'ffn_down'): 'ffn_out-{layer}',
    ('attn', 'post_ffn'): 'l_out-{layer}',
    ('attn', 'q_proj'): None,
    ('attn', 'k_norm'): None,
    ('attn', 'q_norm'): None,
    ('attn', 'q_rope'): None,
    ('attn', 'k_rope'): None,
    ('attn', 'v_proj'): None,
    ('attn', 'attn_raw'): 'attn_pregate-{layer}',
    ('attn', 'attn_gated'): 'attn_gated-{layer}',
    ('attn', 'gate_src'): None,
    ('attn', 'post_ffn'): None,
    ('dbg', 'conv_w'): None,
    ('dbg', 'conv_inp'): None,
}


def build_mappings(our_data, ref_data):
    """Build list of comparable (our_key, ref_name) pairs."""
    mappings = []
    # Reverse lookup: ref_name -> our pattern
    ref_patterns = {}
    for (phase, label), ref_pat in OUR_TO_REF.items():
        if ref_pat:
            ref_patterns[ref_pat] = (phase, label)

    # Scan our data for matches
    for (phase, step, label), vals in our_data.items():
        pat = OUR_TO_REF.get((phase, label))
        if not pat:
            continue
        ref_name = pat.format(layer=step)
        if ref_name in ref_data:
            mappings.append(((phase, step, label), ref_name))

    return mappings


# =============================================================================
# STATISTICS
# =============================================================================

def compare_tensors(our_vals, ref_vals):
    """Return dict of comparison statistics."""
    n = len(our_vals)
    if n != len(ref_vals):
        return {'error': f'size mismatch: {n} vs {len(ref_vals)}'}
    if n == 0:
        return {'error': 'empty tensor'}

    diffs = [abs(our_vals[i] - ref_vals[i]) for i in range(n)]
    rels = []
    for i in range(n):
        denom = max(abs(ref_vals[i]), 1e-8)
        rels.append(diffs[i] / denom)

    mean_ref = sum(abs(v) for v in ref_vals) / n
    max_abs = max(abs(v) for v in ref_vals)

    # Robust relative diff: use max(|ref|, 1e-3 * max_abs) as denominator
    # to avoid blowing up on near-zero values
    robust_rels = []
    min_denom = max(1e-6, max_abs * 1e-3)
    for i in range(n):
        denom = max(abs(ref_vals[i]), min_denom)
        robust_rels.append(diffs[i] / denom)

    stats = {
        'n': n,
        'max_diff': max(diffs),
        'mean_diff': sum(diffs) / n,
        'rel_max': max(robust_rels),
        'rel_mean': sum(robust_rels) / n,
        'rel_95': sorted(robust_rels)[int(n * 0.95)] if n > 1 else robust_rels[0],
        'rel_99': sorted(robust_rels)[int(n * 0.99)] if n > 1 else robust_rels[0],
        'gt_1pct': sum(1 for r in robust_rels if r > 0.01),
        'gt_5pct': sum(1 for r in robust_rels if r > 0.05),
        'gt_10pct': sum(1 for r in robust_rels if r > 0.10),
        'max_abs_ref': max_abs,
    }

    # PASS thresholds:
    #   exact-match kernels (embedding, RMS norm): max_diff < 1e-4
    #   quantized matvec (Q4_K/Q5_K/Q6_K dp4a): max_diff / max_abs_ref < 0.05
    # Normalising by max_abs_ref avoids false FAILs from near-zero denominators
    # (a 0.01 absolute diff on an element whose ref is 0.003 gives 3x relative
    # diff even though the error is negligible relative to the tensor's scale).
    scale_err = stats['max_diff'] / max(max_abs, 1e-6)
    stats['pass'] = (stats['max_diff'] < 1e-4) or (scale_err < 0.05)
    return stats


def corr_coeff(a, b):
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((v - ma) ** 2 for v in a) * sum((v - mb) ** 2 for v in b))
    return num / den if den > 0 else 0.0


# =============================================================================
# REPORTING
# =============================================================================

def print_header(title):
    print()
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_tensor_table(results):
    print_header("TENSOR-BY-TENSOR COMPARISON (using LAST reference occurrence)")
    print()
    hdr = f"{'Our Label':<30} {'Ref Label':<30} {'N':>7} {'MaxDiff':>10} {'RelMax':>8} {'RelMean':>8} {'>1%':>5} {'>5%':>5} {'Status':>6}"
    print(hdr)
    print("-" * len(hdr))

    first_fail = None
    for our_key, ref_name, stats in results:
        if 'error' in stats:
            print(f"{' '.join(map(str, our_key)):<30} {ref_name:<30} ERROR: {stats['error']}")
            continue
        label = our_key[2]
        status = "PASS" if stats['pass'] else "FAIL"
        if not stats['pass'] and first_fail is None:
            first_fail = (our_key, ref_name, stats)
        print(f"{label:<30} {ref_name:<30} {stats['n']:>7} {stats['max_diff']:>10.6f} "
              f"{stats['rel_max']:>8.4f} {stats['rel_mean']:>8.4f} {stats['gt_1pct']:>5} {stats['gt_5pct']:>5} {status:>6}")

    print()
    if first_fail:
        print(f"  First divergence: {first_fail[0][2]} vs {first_fail[1]}  max_diff={first_fail[2]['max_diff']:.6g}")
    else:
        print("  All compared tensors PASS within tolerance.")
    print()


def print_summary(results, e2e=None):
    print_header("SUMMARY")
    total = len(results)
    passed = sum(1 for _, _, s in results if s.get('pass', False))
    print(f"  Tensors compared: {total}")
    print(f"  Tensors PASS:     {passed}")
    print(f"  Tensors FAIL:     {total - passed}")
    print()

    if e2e:
        print(f"  E2E Correlation:  {e2e['corr']:.6f}")
        print(f"  E2E MSE:          {e2e['mse']:.4f}")
        print(f"  Ref top1 token:   {e2e['ref_top1']}")
        print(f"  Our top1 token:   {e2e['our_top1']}")
        print()


def load_logits_bin(path, n=248320):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read(n * 4)
    if len(data) != n * 4:
        return None
    return struct.unpack(f'<{n}f', data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Comprehensive mono27b vs llama.cpp parity checker")
    ap.add_argument('--our-debug', default='/tmp/mono_verify.debug.tsv', help='Our debug TSV')
    ap.add_argument('--ref', default='debug/verify/ref_intermediates.txt', help='llama-debug stdout')
    ap.add_argument('--ref-logits', default='debug/verify/ref_logits.bin', help='Reference logits binary')
    ap.add_argument('--our-logits', default='debug/verify/our_logits.bin', help='Our logits binary')
    args = ap.parse_args()

    # Resolve relative paths
    script_dir = Path(__file__).resolve().parent
    for attr in ['ref', 'ref_logits', 'our_logits']:
        p = getattr(args, attr)
        if not os.path.isabs(p):
            setattr(args, attr, str(script_dir / p))

    print("Loading our debug dump...")
    our_data = parse_our_tsv(args.our_debug)
    print(f"  -> {len(our_data)} tensors")

    print("Loading reference output...")
    ref_data = parse_ref_stdout(args.ref)
    print(f"  -> {len(ref_data)} unique tensors (last occurrence kept)")

    mappings = build_mappings(our_data, ref_data)
    print(f"  -> {len(mappings)} comparable tensor pairs")

    results = []
    for our_key, ref_name in mappings:
        stats = compare_tensors(our_data[our_key], ref_data[ref_name])
        results.append((our_key, ref_name, stats))

    # Sort by layer then label for readability
    results.sort(key=lambda x: (x[0][1], x[0][2]))

    print_tensor_table(results)

    # E2E logits
    e2e = None
    ref_l = load_logits_bin(args.ref_logits)
    our_l = load_logits_bin(args.our_logits)
    if ref_l and our_l:
        corr = corr_coeff(ref_l, our_l)
        mse = sum((ref_l[i] - our_l[i]) ** 2 for i in range(len(ref_l))) / len(ref_l)
        ref_top1 = max(range(len(ref_l)), key=lambda i: ref_l[i])
        our_top1 = max(range(len(our_l)), key=lambda i: our_l[i])
        e2e = {'corr': corr, 'mse': mse, 'ref_top1': ref_top1, 'our_top1': our_top1}

    print_summary(results, e2e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
