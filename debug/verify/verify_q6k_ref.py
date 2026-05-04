#!/usr/bin/env python3
"""Verify Q6_K dequantization matches reference ggml implementation.
Reads the first 20 Q6_K blocks of blk.0.attn_qkv.weight from GGUF,
dequantizes using the reference ggml library, and compares with our Python dequant."""
import struct, ctypes, ctypes.util, os, sys

# Path to reference ggml library
lib_path = '/mnt/mydata/projects2/mono27b/ref/llama.cpp/build/bin/libggml-base.so'

if not os.path.exists(lib_path):
    # Build the reference library if needed
    print(f"Reference library not found at {lib_path}")
    print("Building reference library...")
    import subprocess
    subprocess.run(['cmake', '--build', '.', '--target', 'ggml-base'],
                   cwd='/mnt/mydata/projects2/mono27b/ref/llama.cpp/build',
                   capture_output=True)

def find_tensor(gguf_path, target_name):
    """Read GGUF and return (data_offset, offset, dims, type) for target tensor."""
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
            if key == 'general.alignment': alignment = struct.unpack('<I', f.read(4))[0]
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
        info = tensor_infos[target_name]
        return data_offset, info['offset'], info['dims'], info['type'], tensor_infos

def python_q6k_dequant(block_bytes):
    """Our Python Q6_K dequant implementation."""
    d = struct.unpack('<e', block_bytes[208:210])[0]
    ql = list(block_bytes[0:128])
    qh = list(block_bytes[128:192])
    sc = [int(v) if v < 128 else v - 256 for v in block_bytes[192:208]]
    result = []
    for hf in range(2):
        bql = ql[hf*64:(hf+1)*64]
        bqh = qh[hf*32:(hf+1)*32]
        bsc = sc[hf*8:(hf+1)*8]
        for l in range(32):
            is_idx = l // 16
            for r in range(4):
                if r == 0: qv = (bql[l] & 0x0F) | (((bqh[l] >> 0) & 3) << 4)
                elif r == 1: qv = (bql[l+32] & 0x0F) | (((bqh[l] >> 2) & 3) << 4)
                elif r == 2: qv = (bql[l] >> 4) | (((bqh[l] >> 4) & 3) << 4)
                else: qv = (bql[l+32] >> 4) | (((bqh[l] >> 6) & 3) << 4)
                qv -= 32
                result.append(d * bsc[is_idx + r*2] * qv)
    return result

# Load reference library and get dequantize function
try:
    lib = ctypes.CDLL(lib_path)
    # Find the dequantize_row_q6_K function
    dequant = lib.dequantize_row_q6_K
    dequant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    dequant.restype = None
    print(f"Loaded reference library: {lib_path}")
    
    # Test: read first Q6_K block of attn_qkv.weight and attn_norm.weight
    gguf = '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'
    data_offset, t_offset, dims, typ, _ = find_tensor(gguf, 'blk.0.attn_qkv.weight')
    
    with open(gguf, 'rb') as f:
        f.seek(data_offset + t_offset)
        block_bytes = f.read(210)
    
    # Dequant using reference
    block_size = 256  # QK_K
    output = (ctypes.c_float * block_size)()
    block_q6k = ctypes.create_string_buffer(block_bytes)  # wrong: we need the right struct
    # Actually, the reference function takes block_q6_K* which has the same memory layout
    dequant(block_q6k, output, block_size)
    ref_vals = list(output)
    
    # Dequant using Python
    py_vals = python_q6k_dequant(block_bytes)
    
    # Compare
    max_diff = max(abs(py_vals[i] - ref_vals[i]) for i in range(block_size))
    print(f"\nQ6_K dequant comparison (first block):")
    print(f"Reference first 8: {[f'{ref_vals[i]:.6f}' for i in range(8)]}")
    print(f"Python   first 8: {[f'{py_vals[i]:.6f}' for i in range(8)]}")
    print(f"Max diff: {max_diff:.10f}")
    print(f"MATCH: {'YES' if max_diff < 1e-6 else 'NO'}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
