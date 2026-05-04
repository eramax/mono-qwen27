#!/usr/bin/env python3
import re, sys, os, math

# Parse our debug TSV
our_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mono_verify.debug.tsv'
ref_file = sys.argv[2] if len(sys.argv) > 2 else 'ref_intermediates.txt'

our_data = {}
with open(our_file) as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue
        phase, step, pos, tok, label, n = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4], int(parts[5])
        vals = [float(v) for v in parts[10].split(',') if v]
        our_data[(phase, step, label)] = vals

# Parse reference file - use LAST occurrence of each tensor
ref_data = {}
with open(ref_file) as f:
    content = f.read()

blocks = re.split(r'common_debug_cb_eval:\s+', content)
for block in blocks:
    block = block.strip()
    if not block:
        continue
    # Extract tensor name
    m = re.match(r'(\S+)\s+=\s+\(f32\)', block)
    if not m:
        continue
    name = m.group(1)
    # Extract array content between first [ and matching ]
    arr_match = re.search(r'(\[.*?\])\s*\n\s*sum\s+=', block, re.DOTALL)
    if not arr_match:
        continue
    arr_str = arr_match.group(1)
    # Flatten nested brackets to floats
    flat = arr_str.replace('[',' ').replace(']',' ').replace(',',' ')
    try:
        vals = [float(x) for x in flat.split() if x.strip()]
    except ValueError:
        continue
    # Store as last occurrence (overwrite)
    ref_data[name] = vals

# Mapping from our labels to ref labels for layer 0, step 0 (prompt)
label_map = {
    ('embed', 0, 'h'): 'model.input_embed',
    ('ssm', 0, 'attn_norm'): 'attn_norm-0',
    ('ssm', 0, 'z'): 'z-0',
    ('ssm', 0, 'conv_raw'): 'conv_output_raw-0',
    ('ssm', 0, 'conv'): 'conv_output_silu-0',
    ('ssm', 0, 'q_conv_predelta'): 'q_conv_predelta-0',
    ('ssm', 0, 'k_conv_predelta'): 'k_conv_predelta-0',
    ('ssm', 0, 'deltanet'): 'attn_output-0',
    ('ssm', 0, 'final_output'): 'final_output-0',
    ('ssm', 0, 'post_norm'): 'post_attention_norm-0',
    ('ssm', 0, 'ffn_gate'): 'ffn_gate-0',
    ('ssm', 0, 'ffn_up'): 'ffn_up-0',
    ('ssm', 0, 'ffn_down'): 'ffn_down-0',
    ('ssm', 0, 'layer_out'): 'l_out-0',
}

print("Comparison of prompt step (step=0) layer 0 — using LAST reference occurrence:")
print(f"{'Our Label':<40} {'Ref Label':<40} {'Len':>6} {'MaxDiff':>12} {'Status':>8}")
print("-" * 110)

first_diff_label = None
for key, ref_name in label_map.items():
    if ref_name is None:
        continue
    if key not in our_data:
        print(f"{' '.join(map(str,key)):<40} {ref_name:<40} MISSING in our data")
        continue
    if ref_name not in ref_data:
        print(f"{' '.join(map(str,key)):<40} {ref_name:<40} MISSING in ref data")
        continue
    our_vals = our_data[key]
    ref_vals = ref_data[ref_name]
    if len(our_vals) != len(ref_vals):
        print(f"{' '.join(map(str,key)):<40} {ref_name:<40} {'':>6} LENGTH MISMATCH {len(our_vals)} vs {len(ref_vals)}")
        continue
    max_diff = max(abs(o - r) for o, r in zip(our_vals, ref_vals))
    status = "PASS" if max_diff < 1e-4 else "DIFF"
    if status == "DIFF" and first_diff_label is None:
        first_diff_label = (key, ref_name, max_diff)
    print(f"{' '.join(map(str,key)):<40} {ref_name:<40} {len(our_vals):>6} {max_diff:>12.6g} {status:>8}")

if first_diff_label:
    print(f"\nFirst divergence: {first_diff_label[0]} vs {first_diff_label[1]}  max_diff={first_diff_label[2]:.6g}")
else:
    print("\nAll checked labels match within tolerance.")
