#include "include/mono27b_gguf.h"
#include <cstdio>
#include <string>
#include <unordered_map>

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    // We need to read the KVs. mono27b_read_gguf doesn't expose them easily.
    // I'll use a custom reader for KVs.
    std::FILE * fp = std::fopen(argv[1], "rb");
    char magic[4];
    std::fread(magic, 4, 1, fp);
    uint32_t version, n_tensors, n_kv;
    std::fread(&version, 4, 1, fp);
    std::fread(&n_tensors, 4, 1, fp);
    std::fread(&n_kv, 4, 1, fp);
    
    for (uint32_t i = 0; i < n_kv; ++i) {
        uint64_t key_len;
        std::fread(&key_len, 8, 1, fp);
        char key[256];
        std::fread(key, key_len, 1, fp);
        key[key_len] = 0;
        uint32_t type;
        std::fread(&type, 4, 1, fp);
        if (std::string(key) == "attention.layer_norm_rms_epsilon") {
            float eps;
            std::fread(&epsilon, 4, 1, fp);
            std::printf("RMS Epsilon: %g\n", eps);
        }
        // skip value
        if (type == 6) { float f; std::fread(&f, 4, 1, fp); }
        else if (type == 4) { uint32_t u; std::fread(&u, 4, 1, fp); }
        // ... this is too hard. I'll use rtk to dump it if I can.
    }
    return 0;
}
