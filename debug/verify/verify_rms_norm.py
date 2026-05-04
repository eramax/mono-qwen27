#!/usr/bin/env python3
"""Verify RMS norm computation matches GPU output.
Usage: python3 verify_rms_norm.py [gguf_path] [debug_path]
"""
import struct, sys, math

def read_tensors(gguf_path):
    with open(gguf_path, 'rb') as f:
        f.read(4); struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        alignment = 32
        for i in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            val_type = struct.unpack('<I', f.read(4))[0]
            if key == 'general.alignment': alignment = struct.unpack('<I', f.read(4))[0]
            elif val_type == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
            elif val_type in (4,5,6): f.read(4)
            elif val_type == 7: f.read(1)
            elif val_type in (10,11,12): f.read(8)
            elif val_type == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]; arr_n = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:
                    for _ in range(arr_n): sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
                else: f.read(4 * arr_n)
            else: f.read(1)
        tensor_infos = {}
        for i in range(n_tensors):
            nl = struct.unpack('<Q', f.read(8))[0]
            nm = f.read(nl).decode('utf-8', errors='replace')
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            typ = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensor_infos[nm] = {'offset': off, 'dims': dims, 'type': typ}
        tensors_end = f.tell()
        data_offset = (tensors_end + alignment - 1) & ~(alignment - 1)
        return data_offset, tensor_infos

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    debug_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mono_v2.debug.tsv'
    
    # Read data from debug
    with open(debug_path, 'r') as f:
        content = f.read()
    
    embed, attn_norm = None, None
    for line in content.split('\n'):
        if line.startswith('embed\t0\t0\t44883\th\t5120'):
            parts = line.strip().split('\t')
            embed = [float(v) for v in parts[10].split(',')]
        if 'ssm\t0\t0\t44883\tattn_norm\t5120' in line:
            parts = line.strip().split('\t')
            attn_norm = [float(v) for v in parts[10].split(',')]
    if not embed or not attn_norm:
        print("ERROR: Could not find embed/attn_norm in debug file"); return 1
    
    print(f"Embed L2: {math.sqrt(sum(v*v for v in embed)):.6f}")
    
    # Read attn_norm.weight from GGUF
    data_offset, tensor_infos = read_tensors(gguf_path)
    info = tensor_infos['blk.0.attn_norm.weight']
    
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + info['offset'])
        weights = struct.unpack(f'<{info["dims"][0]}f', f.read(info['dims'][0] * 4))
    
    # CPU RMS norm computation
    n = len(embed)
    mean_sq = sum(v*v for v in embed) / n
    eps = 1e-6
    inv = 1.0 / math.sqrt(mean_sq + eps)
    print(f"inv: {inv:.6f} (from mean_sq={mean_sq:.10f})")
    
    max_diff, argmax = 0, 0
    for i in range(min(n, len(attn_norm))):
        cpu_val = embed[i] * inv * weights[i]
        gpu_val = attn_norm[i]
        diff = abs(cpu_val - gpu_val)
        if diff > max_diff:
            max_diff = diff
            argmax = i
    
    print(f"RMS norm max diff: {max_diff:.10f} at element {argmax}")
    print(f"MATCH: {'YES' if max_diff < 1e-5 else 'NO'}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
