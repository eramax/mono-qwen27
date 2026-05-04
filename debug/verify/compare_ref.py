#!/usr/bin/env python3
"""Compare GPU debug output with reference llama-debug output for layer 0.
Usage: python3 compare_ref.py [gpu_debug.tsv] [ref_debug.txt]
"""
import re, sys, math

def extract_values(text, tensor_name):
    """Extract tensor values from llama-debug output after DEBUG_VALUES."""
    pattern = re.compile(
        r'DEBUG_VALUES\tcommon_debug_cb_eval\t' + re.escape(tensor_name) + r'\n'
        r'(.*?)(?=DEBUG_MATCH|\Z)', re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    block = match.group(1)
    # Extract all float numbers
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', block)
    return [float(n) for n in nums] if nums else None

if __name__ == '__main__':
    ref_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ref_out.txt'
    gpu_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mono_v2.debug.tsv'
    
    with open(ref_path, 'r') as f:
        ref_text = f.read()
    with open(gpu_path, 'r') as f:
        gpu_text = f.read()
    
    gpu_data = {}
    for line in gpu_text.split('\n'):
        if '\tssm\t0\t0\t44883\t' in line:
            parts = line.strip().split('\t')
            label = parts[4]
            vals = [float(v) for v in parts[10].split(',')]
            gpu_data[label] = vals
        elif line.startswith('embed\t0\t0\t44883\th\t5120'):
            parts = line.strip().split('\t')
            gpu_data['embed'] = [float(v) for v in parts[10].split(',')]
    
    comparisons = [
        ('final_output-0', 'deltanet', 'DeltaNet output (after gated norm)'),
        ('linear_attn_out-0', 'ssm_out', 'SSM output'),
        ('l_out-0', 'post_ffn', 'Layer output'),
        ('result_norm', 'out_output_norm', 'Output norm (wait, not in GPU debug)'),
        ('result_output', 'logits', 'Final logits'),
    ]
    
    for ref_name, gpu_name, desc in comparisons:
        ref_vals = extract_values(ref_text, ref_name)
        gpu_vals = gpu_data.get(gpu_name)
        
        print(f"\n=== {desc} ===")
        print(f"Ref tensor: {ref_name}")
        
        if ref_vals:
            print(f"  Ref: {len(ref_vals)} vals, first 16: {[f'{v:.6f}' for v in ref_vals[:16]]}")
            print(f"  Ref: min={min(ref_vals):.6f} max={max(ref_vals):.6f} L2={math.sqrt(sum(v*v for v in ref_vals)):.6f}")
        else:
            print(f"  Ref: NOT FOUND")
        
        if gpu_vals:
            print(f"  GPU: {len(gpu_vals)} vals, first 16: {[f'{v:.6f}' for v in gpu_vals[:16]]}")
            print(f"  GPU: min={min(gpu_vals):.6f} max={max(gpu_vals):.6f} L2={math.sqrt(sum(v*v for v in gpu_vals)):.6f}")
        else:
            print(f"  GPU: NOT FOUND in debug")
        
        if ref_vals and gpu_vals and len(ref_vals) == len(gpu_vals):
            max_diff = max(abs(ref_vals[i] - gpu_vals[i]) for i in range(len(ref_vals)))
            print(f"  Max diff: {max_diff:.6f}")
        elif ref_vals and gpu_vals and len(ref_vals) != len(gpu_vals):
            print(f"  SIZE MISMATCH: ref={len(ref_vals)} gpu={len(gpu_vals)}")
    
    # Extract GPU logits
    for line in gpu_text.split('\n'):
        if 'out\t-1\t0\t44883\tlogits\t248320' in line:
            parts = line.strip().split('\t')
            gpu_logits = [float(v) for v in parts[10].split(',')]
            print(f"\n=== GPU Logits ===")
            print(f"First 16: {[f'{v:.4f}' for v in gpu_logits[:16]]}")
            top1 = max(range(len(gpu_logits)), key=lambda i: gpu_logits[i])
            print(f"Top1: {top1} (logit={gpu_logits[top1]:.4f})")
            break
