#!/usr/bin/env python3
"""Verify Q6_K matvec for wqkv (attn_qkv.weight) against ggml reference dequant."""
import struct

def dequant_q6k_block(block_bytes):
    """Dequantize one Q6_K block (210 bytes) into 256 floats.
    
    Reference: dequantize_row_q6_K in ggml-quants.c
    """
    d = struct.unpack('<e', block_bytes[208:210])[0]
    ql = list(block_bytes[0:128])
    qh = list(block_bytes[128:192])
    sc = [int(v) for v in block_bytes[192:208]]  # int8

    result = []
    for hf in range(2):  # 2 halves of 128 elements
        bql = ql[hf*64:(hf+1)*64]
        bqh = qh[hf*32:(hf+1)*32]
        bsc = sc[hf*8:(hf+1)*8]
        for l in range(32):
            is_idx = l // 16
            for r in range(4):
                if r == 0:
                    qv = (bql[l] & 0x0F) | (((bqh[l] >> 0) & 3) << 4)
                elif r == 1:
                    qv = (bql[l+32] & 0x0F) | (((bqh[l] >> 2) & 3) << 4)
                elif r == 2:
                    qv = (bql[l] >> 4) | (((bqh[l] >> 4) & 3) << 4)
                else:
                    qv = (bql[l+32] >> 4) | (((bqh[l] >> 6) & 3) << 4)
                qv -= 32
                sc_idx = is_idx + r * 2
                result.append(d * bsc[sc_idx] * qv)
    return result

def read_all_tensors(gguf_path):
    """Read GGUF file, return (data_offset, tensor_infos dict)."""
    with open(gguf_path, 'rb') as f:
        f.read(4)
        struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        alignment = 32
        for i in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            val_type = struct.unpack('<I', f.read(4))[0]
            if key == 'general.alignment':
                alignment = struct.unpack('<I', f.read(4))[0]
            elif val_type == 8:
                sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
            elif val_type in (4,5,6): f.read(4)
            elif val_type == 7: f.read(1)
            elif val_type in (10,11,12): f.read(8)
            elif val_type == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_n = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:
                    for _ in range(arr_n):
                        sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
                else:
                    f.read(4 * arr_n)
            else:
                f.read(1)
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
    import sys
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    
    data_offset, tensor_infos = read_all_tensors(gguf_path)
    
    # Test output.weight (LM head) - known correct via earlier verification
    outw = tensor_infos['output.weight']
    print(f"output.weight: offset={outw['offset']}, dims={outw['dims']}, type={outw['type']}")
    
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + outw['offset'])
        lm_block = f.read(210)
        lm_vals = dequant_q6k_block(lm_block)
        print(f"LM head row 0 first 8: {[f'{v:.6f}' for v in lm_vals[:8]]}")
    
    # Test attn_qkv.weight (wqkv)
    atkv = tensor_infos['blk.0.attn_qkv.weight']
    print(f"\nattn_qkv.weight: offset={atkv['offset']}, dims={atkv['dims']}, type={atkv['type']}")
    
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + atkv['offset'])
        qkv_block = f.read(210)
        qkv_vals = dequant_q6k_block(qkv_block)
        print(f"attn_qkv row 0 first 8: {[f'{v:.6f}' for v in qkv_vals[:8]]}")
    
    # Check d values
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + outw['offset'] + 208)
        lm_d = struct.unpack('<e', f.read(2))[0]
        f.seek(data_offset + atkv['offset'] + 208)
        qkv_d = struct.unpack('<e', f.read(2))[0]
        print(f"\nLM head  first block d: {lm_d:.8f}")
        print(f"attn_qkv first block d: {qkv_d:.8f}")
        print(f"Both reasonable: {'YES' if (1e-6 < lm_d < 100 and 1e-6 < qkv_d < 100) else 'NO'}")
    
    # Now verify against debug dump
    # Read h2 from debug file
    debug_path = '/tmp/mono_v2.debug.tsv'
    try:
        with open(debug_path, 'r') as f:
            content = f.read()
        
        h2 = None
        wqkv_gpu = None
        for line in content.split('\n'):
            if 'attn_norm\t5120' in line:
                parts = line.strip().split('\t')
                h2 = [float(v) for v in parts[10].split(',')]
            if 'ssm\t0\t0\t44883\twqkv\t10240' in line and 'wqkv_gate' not in line:
                parts = line.strip().split('\t')
                wqkv_gpu = [float(v) for v in parts[10].split(',')]
        
        if h2 and wqkv_gpu:
            # Compute full row 0 dot
            with open(gguf_path, 'rb') as f:
                f.seek(data_offset + atkv['offset'])
                row0_vals = []
                for b in range(20):  # 20 blocks = 5120 elements
                    row0_vals.extend(dequant_q6k_block(f.read(210)))
            
            cpu_dot = sum(row0_vals[i] * h2[i] for i in range(5120))
            print(f"\n=== Verification ===")
            print(f"CPU dot row 0: {cpu_dot:.8f}")
            print(f"GPU wqkv[0]:   {wqkv_gpu[0]:.8f}")
            print(f"Diff: {abs(cpu_dot - wqkv_gpu[0]):.8f}")
            print(f"MATCH: {'YES' if abs(cpu_dot - wqkv_gpu[0]) < 1e-3 else 'NO'}")
        else:
            print(f"\nCould not find debug data: h2={'found' if h2 else 'missing'}, wqkv_gpu={'found' if wqkv_gpu else 'missing'}")
    except FileNotFoundError:
        print(f"\nDebug file not found: {debug_path}")
        print("Run mono27b_chat first to generate debug data")
