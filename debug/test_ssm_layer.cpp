// Test: run one SSM layer using ggml and compare with mono27b debug output
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cinttypes>

// Minimal GGUF reader
struct TensorInfo {
    std::string name;
    int n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t offset;
    uint64_t size_bytes;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }
    
    // Read GGUF headers
    FILE * fp = fopen(argv[1], "rb");
    if (!fp) { perror("fopen"); return 1; }
    
    char magic[4];
    uint32_t version, n_tensors, n_kv;
    fread(magic, 4, 1, fp);
    fread(&version, 4, 1, fp);
    fread(&n_tensors, 4, 1, fp);
    fread(&n_kv, 4, 1, fp);
    
    printf("GGUF: tensors=%u kv=%u\n", n_tensors, n_kv);
    
    // Skip KV pairs
    for (uint32_t i = 0; i < n_kv; i++) {
        uint64_t key_len;
        fread(&key_len, 8, 1, fp);
        fseek(fp, key_len, SEEK_CUR);
        uint32_t val_type;
        fread(&val_type, 4, 1, fp);
        if (val_type == 8) { // string
            uint64_t slen;
            fread(&slen, 8, 1, fp);
            fseek(fp, slen, SEEK_CUR);
        } else if (val_type == 4 || val_type == 5 || val_type == 6) {
            fseek(fp, 4, SEEK_CUR);
        } else if (val_type == 7) {
            fseek(fp, 1, SEEK_CUR);
        } else if (val_type == 10 || val_type == 11 || val_type == 12) {
            fseek(fp, 8, SEEK_CUR);
        } else if (val_type == 9) { // array
            uint32_t arr_type;
            uint64_t arr_n;
            fread(&arr_type, 4, 1, fp);
            fread(&arr_n, 8, 1, fp);
            if (arr_type == 8) {
                for (uint64_t j = 0; j < arr_n; j++) {
                    uint64_t slen;
                    fread(&slen, 8, 1, fp);
                    fseek(fp, slen, SEEK_CUR);
                }
            } else {
                fseek(fp, 4 * arr_n, SEEK_CUR);
            }
        } else {
            printf("Unknown KV type %d\n", val_type);
            fseek(fp, 1, SEEK_CUR);
        }
    }
    
    long meta_end = ftell(fp);
    uint32_t alignment = 32;
    
    // Read tensor info
    std::vector<TensorInfo> tensors;
    tensors.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        TensorInfo ti;
        uint64_t name_len;
        fread(&name_len, 8, 1, fp);
        ti.name.resize(name_len);
        fread(&ti.name[0], name_len, 1, fp);
        fread(&ti.n_dims, 4, 1, fp);
        uint64_t nelements = 1;
        for (int d = 0; d < ti.n_dims; d++) {
            fread(&ti.dims[d], 8, 1, fp);
            nelements *= ti.dims[d];
        }
        fread(&ti.ggml_type, 4, 1, fp);
        fread(&ti.offset, 8, 1, fp);
        
        // Compute size_bytes
        static const int type_size[] = {4, 2, 0, 0, 0, 0, 0, 0, 34, 0, 80, 110, 144, 176, 210, 0};
        static const int block_size[] = {1, 1, 0, 0, 0, 0, 0, 0, 32, 0, 256, 256, 256, 256, 256, 0};
        int ts = ti.ggml_type <= 15 ? type_size[ti.ggml_type] : (ti.ggml_type == 23 ? 136 : 0);
        int bs = ti.ggml_type <= 15 ? block_size[ti.ggml_type] : (ti.ggml_type == 23 ? 256 : 0);
        if (ts > 0 && bs > 0 && nelements % bs == 0) {
            ti.size_bytes = (nelements / bs) * ts;
        } else {
            ti.size_bytes = 0;
        }
        tensors.push_back(ti);
    }
    
    uint64_t data_offset = ((uint64_t)meta_end + alignment - 1) & ~(alignment - 1);
    printf("Data offset: %lu\n", (unsigned long)data_offset);
    
    // Find blk.0.ssm_norm.weight and other weights
    for (auto & t : tensors) {
        if (t.name == "blk.0.attn_norm.weight" || 
            t.name == "blk.0.ssm_norm.weight" ||
            t.name == "output_norm.weight" ||
            t.name == "output.weight") {
            printf("Tensor: %s, type=%u, dims=[%lu,%lu,%lu,%lu], offset=%lu, size=%lu\n",
                   t.name.c_str(), t.ggml_type,
                   (unsigned long)t.dims[0], (unsigned long)t.dims[1],
                   (unsigned long)t.dims[2], (unsigned long)t.dims[3],
                   (unsigned long)t.offset, (unsigned long)t.size_bytes);
        }
    }
    
    // Read attn_norm.weight for verification
    for (auto & t : tensors) {
        if (t.name == "blk.0.attn_norm.weight") {
            std::vector<float> w(t.size_bytes / 4);
            fseek(fp, data_offset + t.offset, SEEK_SET);
            fread(w.data(), 1, t.size_bytes, fp);
            printf("attn_norm.weight[0..7]: ");
            for (int i = 0; i < 8 && i < (int)w.size(); i++)
                printf("%.6f ", w[i]);
            printf("\n");
        }
    }
    
    fclose(fp);
    printf("Done.\n");
    return 0;
}
