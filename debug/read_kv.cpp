#include "include/mono27b_gguf.h"
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

struct KvEntry {
    uint32_t type;
    union {
        uint32_t u32;
        float f32;
        uint64_t u64;
    };
    std::string str;
};

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    std::FILE * fp = std::fopen(argv[1], "rb");
    char magic[4]; std::fread(magic, 4, 1, fp);
    uint32_t version, n_tensors, n_kv;
    std::fread(&version, 4, 1, fp);
    std::fread(&n_tensors, 4, 1, fp);
    std::fread(&n_kv, 4, 1, fp);
    
    for (uint32_t i = 0; i < n_kv; ++i) {
        uint64_t key_len; std::fread(&key_len, 8, 1, fp);
        std::vector<char> key_buf(key_len + 1);
        std::fread(key_buf.data(), key_len, 1, fp);
        key_buf[key_len] = 0;
        std::string key(key_buf.data());
        uint32_t type; std::fread(&type, 4, 1, fp);
        if (type == 6) {
            float f; std::fread(&f, 4, 1, fp);
            if (key.find("epsilon") != std::string::npos) std::printf("%s: %g\n", key.c_str(), f);
        } else if (type == 12) {
            double f; std::fread(&f, 8, 1, fp);
            if (key.find("epsilon") != std::string::npos) std::printf("%s: %g\n", key.c_str(), f);
        } else if (type == 4) {
            uint32_t u; std::fread(&u, 4, 1, fp);
        } else if (type == 10 || type == 11) {
            uint64_t u; std::fread(&u, 8, 1, fp);
        } else if (type == 8) {
            uint64_t slen; std::fread(&slen, 8, 1, fp);
            std::fseek(fp, slen, SEEK_CUR);
        } else if (type == 9) {
            uint32_t at; std::fread(&at, 4, 1, fp);
            uint64_t an; std::fread(&an, 8, 1, fp);
            if (at == 8) {
                for (uint64_t j = 0; j < an; ++j) {
                    uint64_t slen; std::fread(&slen, 8, 1, fp);
                    std::fseek(fp, slen, SEEK_CUR);
                }
            } else {
                size_t es = (at == 6 || at == 4 || at == 5) ? 4 : (at == 10 || at == 11 || at == 12) ? 8 : 1;
                std::fseek(fp, es * an, SEEK_CUR);
            }
        } else {
            std::fseek(fp, 1, SEEK_CUR); // hack
        }
    }
    std::fclose(fp);
    return 0;
}
