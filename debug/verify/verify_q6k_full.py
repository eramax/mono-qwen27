#!/usr/bin/env python3
"""Verify Q6_K matvec: read attn_qkv.weight from GGUF, dequantize row 0,
compute dot with h2, compare with GPU wqkv output.
Usage: python3 verify_q6k_full.py [gguf_path] [h2_path] [debug_path]
"""
import struct, math, sys

BLOCK_BYTES = 210

def dequant_q6k_block(block_bytes):
    d = struct.unpack('<e', block_bytes[208:210])[0]
    ql = list(block_bytes[0:128])
    qh = list(block_bytes[128:192])
    sc = [int(v) if v < 128 else v - 256 for v in block_bytes[192:208]]
    vals = []
    for hf in range(2):
        bql = ql[hf*64:(hf+1)*64]
        bqh = qh[hf*32:(hf+1)*32]
        bsc = sc[hf*8:(hf+1)*8]
        for r in range(4):
            for l in range(32):
                is_idx = l // 16
                if r == 0: qv = (bql[l] & 0x0F) | (((bqh[l] >> 0) & 3) << 4)
                elif r == 1: qv = (bql[l+32] & 0x0F) | (((bqh[l] >> 2) & 3) << 4)
                elif r == 2: qv = (bql[l] >> 4) | (((bqh[l] >> 4) & 3) << 4)
                else: qv = (bql[l+32] >> 4) | (((bqh[l] >> 6) & 3) << 4)
                qv -= 32
                vals.append(d * bsc[is_idx + r*2] * qv)
    return vals

def read_gguf(gguf_path):
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

if __name__ == '__main__':
    gguf = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    h2_path = sys.argv[2] if len(sys.argv) > 2 else 'debug/verify/attn_norm_full.txt'
    debug_path = sys.argv[3] if len(sys.argv) > 3 else '/tmp/mono_v2.debug.tsv'
    
    data_off, tensors = read_gguf(gguf)
    atkv = tensors['blk.0.attn_qkv.weight']
    print(f"Q6_K attn_qkv: offset={atkv['offset']}, dims={atkv['dims']}")
    
    with open(h2_path, 'r') as f:
        h2 = [float(line.strip()) for line in f]
    print(f"h2: {len(h2)} elems, L2={math.sqrt(sum(v*v for v in h2)):.6f}")
    
    # Read row 0
    with open(gguf, 'rb') as f:
        f.seek(data_off + atkv['offset'])
        rv = []
        for b in range(20):
            rv.extend(dequant_q6k_block(f.read(BLOCK_BYTES)))
    
    print(f"Row0: {len(rv)} vals, first 8: {[f'{v:.6f}' for v in rv[:8]]}")
    cpu_dot = sum(rv[i] * h2[i] for i in range(5120))
    print(f"CPU dot: {cpu_dot:.8f}")
    
    # Compare with GPU
    gpu0 = None
    with open(debug_path, 'r') as f:
        for line in f:
            if 'ssm\t0\t0\t44883\twqkv\t10240' in line and 'wqkv_gate' not in line:
                parts = line.strip().split('\t')
                gpu0 = float(parts[10].split(',')[0])
    if gpu0 is not None:
        print(f"GPU [0]: {gpu0:.8f}, diff: {abs(cpu_dot - gpu0):.8f}")
    
    # Also verify: what if dim ordering is [ne1, ne0] instead of [ne0, ne1]?
    # Try reading 20 blocks at offset 0 (row 0 of dims[0]) vs 5120/256=20 blocks
    # Actually row_blocks = dims[0]/256. For dims=[5120,10240], row_blocks=20.
    # Row stride = 20 * 210 = 4200 bytes.
    # Row 0: 0..4199. Row 1: 4200..8399. etc.
    # But what if the actual stride is dims[1]/256? = 10240/256 = 40 blocks per row?
    # This would mean each row has 40 blocks = 10240 elements, not 5120!
    
    # Check different row indexing
    with open(gguf, 'rb') as f:
        f.seek(data_off + atkv['offset'])
        # Try reading 40 blocks instead of 20
        rv40 = []
        for b in range(40):
            rv40.extend(dequant_q6k_block(f.read(BLOCK_BYTES)))
    
    # If row has 10240 elements (40 blocks), compute dot with first 5120
    cpu_dot40 = sum(rv40[i] * h2[i] for i in range(5120))
    print(f"CPU with 40 blocks, dot first 5120: {cpu_dot40:.8f}")
    
    # Also check: what if row strides are different?
    # Maybe the weight matrix is transposed: [10240, 5120] instead of [5120, 10240]
    # Then row 0 has 10240 elements (40 blocks), and we only use first 5120 of them
    
    # OR: maybe the matvec uses dims[1] for input dim and dims[0] for output dim
    # In that case, there's only 5120 rows of 10240 elements each
    # And our code reading 20 blocks is wrong

