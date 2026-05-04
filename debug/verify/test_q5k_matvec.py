#!/usr/bin/env python3
"""Verify Q5_K matvec (wqkv_gate) numerically against GPU output.
Usage: python3 debug/verify/test_q5k_matvec.py [gguf_path] [debug_path]
"""
import struct, sys, os

BLOCK_SIZE = 256
BLOCK_BYTES = 176  # Q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)

def get_scale_min_k4(j, scales):
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m

def dequant_q5k_block(block_data, output):
    d = struct.unpack('<e', block_data[0:2])[0]
    dmin = struct.unpack('<e', block_data[2:4])[0]
    scales = list(block_data[4:16])
    qh = list(block_data[16:48])
    qs = list(block_data[48:176])
    for e in range(256):
        n64 = e // 64
        l = e % 32
        hi = (e % 64) // 32
        sc, m = get_scale_min_k4(n64 * 2 + hi, scales)
        qsrc = qs[n64 * 32 + l]
        qv = (qsrc & 0x0F if hi == 0 else qsrc >> 4) + ((qh[l] & (1 << (n64 * 2 + hi))) != 0) * 16
        output.append(d * sc * qv - dmin * m)

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
    
    # Read h2 and wqkv_gate from debug
    with open(debug_path, 'r') as f:
        content = f.read()
    h2, wqkv_gate = None, None
    for line in content.split('\n'):
        if 'ssm\t0\t0\t44883\tattn_norm\t5120' in line:
            parts = line.strip().split('\t')
            h2 = [float(v) for v in parts[10].split(',')]
        if 'ssm\t0\t0\t44883\twqkv_gate' in line:
            parts = line.strip().split('\t')
            wqkv_gate = [float(v) for v in parts[10].split(',')]
    if not h2 or not wqkv_gate:
        print("ERROR: Could not find data in debug file"); return 1
    
    # Read row 0 of attn_gate.weight (wqkv_gate)
    data_offset, tensor_infos = read_tensors(gguf_path)
    info = tensor_infos['blk.0.attn_gate.weight']
    row_elems = info['dims'][0]
    n_blocks = row_elems // BLOCK_SIZE
    
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + info['offset'])
        row0_vals = []
        for b in range(n_blocks):
            dequant_q5k_block(f.read(BLOCK_BYTES), row0_vals)
    
    cpu_dot = sum(row0_vals[i] * h2[i] for i in range(len(row0_vals)))
    gpu_val = wqkv_gate[0]
    print(f"CPU dot: {cpu_dot:.10f}")
    print(f"GPU val: {gpu_val:.10f}")
    print(f"Diff:    {abs(cpu_dot - gpu_val):.10f}")
    print(f"MATCH:   {'YES' if abs(cpu_dot - gpu_val) < 1e-4 else 'NO'}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
