#!/usr/bin/env python3
"""Quick sanity check for Q6_K format: verify d values are in valid range."""
import struct, sys

gguf_path = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf'

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

    # Check output.weight d
    outw = tensor_infos['output.weight']
    f.seek(data_offset + outw['offset'] + 208)
    d = struct.unpack('<e', f.read(2))[0]
    assert 1e-7 < d < 1.0, f'output.weight d={d} out of range'
    print(f'  LM head (Q6_K): first block d={d:.8f} OK')

    # Check attn_qkv.weight d
    atkv = tensor_infos['blk.0.attn_qkv.weight']
    f.seek(data_offset + atkv['offset'] + 208)
    d2 = struct.unpack('<e', f.read(2))[0]
    assert 1e-7 < d2 < 1.0, f'attn_qkv d={d2} out of range'
    print(f'  wqkv (Q6_K): first block d={d2:.8f} OK')

print('All Q6_K sanity checks PASSED')
