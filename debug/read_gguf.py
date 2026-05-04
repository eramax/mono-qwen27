import sys
import struct

def read_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print("Not a GGUF file")
            return
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        for i in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 6: # F32
                val = struct.unpack('<f', f.read(4))[0]
                if 'epsilon' in key: print(f"{key}: {val}")
            elif vtype == 4: # U32
                val = struct.unpack('<I', f.read(4))[0]
            elif vtype == 8: # String
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode('utf-8')
            elif vtype == 9: # Array
                atype = struct.unpack('<I', f.read(4))[0]
                an = struct.unpack('<Q', f.read(8))[0]
                if atype == 8: # String array
                    for j in range(an):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.read(slen)
                else:
                    es = 4 if atype in [4, 5, 6] else 8 if atype in [10, 11, 12] else 1
                    f.read(es * an)
            elif vtype in [0, 1, 2, 3, 4, 5, 7]: # i8, u8, i16, u16, u32, i32, bool
                size = 1 if vtype in [0, 1, 7] else 2 if vtype in [2, 3] else 4
                f.read(size)
            elif vtype in [10, 11, 12]: # u64, i64, f64
                f.read(8)
            else:
                print(f"Unknown type {vtype} for key {key}")
                break

if __name__ == "__main__":
    read_gguf(sys.argv[1])
