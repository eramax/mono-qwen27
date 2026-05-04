// Compile & run:
// g++ -std=c++11 -I ref/llama.cpp/ggml/include -I ref/llama.cpp/ggml/src \
//   -L ref/llama.cpp/build/bin -lggml-base -lpthread -ldl -lm \
//   -o /tmp/ssm_ref_test /tmp/ssm_ref_test.cpp
// LD_LIBRARY_PATH=ref/llama.cpp/build/bin /tmp/ssm_ref_test

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include "ggml.h"
#include "ggml-common.h"

// ---------------------------------------------------------------------------
// Minimal structs matching our CUDA executor
// ---------------------------------------------------------------------------
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    ggml_fp16_t d;
};

struct BlockQ5K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

static float q5k_val(const BlockQ5K & qb, int e) {
    int n64 = e / 64, l = e % 32, hi = (e % 64) / 32;
    uint8_t sc, m;
    get_scale_min_k4(n64 * 2 + hi, qb.scales, &sc, &m);
    float d = ggml_fp16_to_fp32(qb.d);
    float mn = ggml_fp16_to_fp32(qb.dmin);
    uint8_t qsrc = qb.qs[n64 * 32 + l];
    int qv = (hi == 0 ? (qsrc & 0x0F) : (qsrc >> 4)) +
             ((qb.qh[l] & (1 << (n64 * 2 + hi))) ? 16 : 0);
    return d * sc * qv - mn * m;
}

static float q6k_val(const BlockQ6K & qb, int e) {
    int n128 = e / 128, l = e % 32, r = (e % 128) / 32;
    const uint8_t * ql = qb.ql + n128 * 64;
    const uint8_t * qh = qb.qh + n128 * 32;
    const int8_t * sc = qb.scales + n128 * 8;
    int qv;
    if (r == 0) qv = (ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4);
    else if (r == 1) qv = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4);
    else if (r == 2) qv = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    else qv = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
    qv -= 32;
    int sc_idx = (l / 16) + (r * 2);
    return ggml_fp16_to_fp32(qb.d) * (float)sc[sc_idx] * (float)qv;
}

// ---------------------------------------------------------------------------
// Read tensor data from GGUF
// ---------------------------------------------------------------------------
struct GGUFReader {
    FILE * fp;
    long data_offset;
    long tensor_offset;
    int n_blocks;     // blocks per row
    int n_rows;
    int block_size;   // elements per block (256 for K-quants)
    int type_size;    // bytes per block (210 for Q6_K, 176 for Q5_K)
    int ggml_type;
};

GGUFReader open_gguf_tensor(const char * gguf_path, const char * tensor_name, int type_sz, int blk_sz) {
    GGUFReader r = {0};
    FILE * fp = fopen(gguf_path, "rb");
    if (!fp) { fprintf(stderr, "Can't open %s\n", gguf_path); return r; }
    r.fp = fp;
    
    // Read header
    char magic[4]; fread(magic, 1, 4, fp);
    int version; fread(&version, 4, 1, fp);
    long n_tensors, n_kv;
    fread(&n_tensors, 8, 1, fp); fread(&n_kv, 8, 1, fp);
    
    int alignment = 32;
    for (long i = 0; i < n_kv; i++) {
        long key_len; fread(&key_len, 8, 1, fp);
        char * key = new char[key_len+1]; fread(key, 1, key_len, fp); key[key_len]=0;
        int val_type; fread(&val_type, 4, 1, fp);
        if (!strcmp(key, "general.alignment")) { fread(&alignment, 4, 1, fp); }
        else if (val_type == 8) { long sl; fread(&sl, 8, 1, fp); fseek(fp, sl, SEEK_CUR); }
        else if (val_type == 4 || val_type == 5 || val_type == 6) fseek(fp, 4, SEEK_CUR);
        else if (val_type == 7) fseek(fp, 1, SEEK_CUR);
        else if (val_type == 10 || val_type == 11 || val_type == 12) fseek(fp, 8, SEEK_CUR);
        else if (val_type == 9) { int at; long an; fread(&at, 4, 1, fp); fread(&an, 8, 1, fp); fseek(fp, an*4, SEEK_CUR); }
        else fseek(fp, 4, SEEK_CUR);
        delete[] key;
    }
    
    long kv_end = ftell(fp);
    long target_offset = -1, target_ne0 = 0, target_ne1 = 0;
    for (long i = 0; i < n_tensors; i++) {
        long nl; fread(&nl, 8, 1, fp);
        char * nm = new char[nl+1]; fread(nm, 1, nl, fp); nm[nl]=0;
        int nd; fread(&nd, 4, 1, fp);
        long ne0=1, ne1=1;
        if (nd > 0) fread(&ne0, 8, 1, fp);
        if (nd > 1) fread(&ne1, 8, 1, fp);
        int typ; fread(&typ, 4, 1, fp);
        long off; fread(&off, 8, 1, fp);
        if (!strcmp(nm, tensor_name)) { target_offset = off; target_ne0 = ne0; target_ne1 = ne1; }
        delete[] nm;
    }
    
    long tensors_end = ftell(fp);
    r.data_offset = (tensors_end + alignment - 1) & ~(alignment - 1);
    r.tensor_offset = target_offset;
    r.block_size = blk_sz;
    r.type_size = type_sz;
    r.n_blocks = target_ne0 / blk_sz;
    r.n_rows = target_ne1;
    return r;
}

// ---------------------------------------------------------------------------
// Compute one row of a quantized matvec on CPU
// ---------------------------------------------------------------------------
static float q5k_row_dot(const BlockQ5K * row_blocks, int rb, const float * x) {
    float sum = 0;
    for (int b = 0; b < rb; b++) {
        const BlockQ5K & qb = row_blocks[b];
        for (int e = 0; e < 256; e++) sum += q5k_val(qb, e) * x[b * 256 + e];
    }
    return sum;
}

static float q6k_row_dot(const BlockQ6K * row_blocks, int rb, const float * x) {
    float sum = 0;
    for (int b = 0; b < rb; b++) {
        const BlockQ6K & qb = row_blocks[b];
        for (int e = 0; e < 256; e++) sum += q6k_val(qb, e) * x[b * 256 + e];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Main verification
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : 
        "/home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf";
    const char * h2_path = argc > 2 ? argv[2] : "/tmp/h2_norm_full.txt";
    
    // Read h2 from file
    float h2[5120];
    FILE * hfp = fopen(h2_path, "r");
    if (!hfp) { fprintf(stderr, "open h2 file\n"); return 1; }
    for (int i = 0; i < 5120; i++) {
        if (fscanf(hfp, "%f", &h2[i]) != 1) break;
    }
    fclose(hfp);
    printf("Loaded h2: first=%.8f last=%.8f\n", h2[0], h2[5119]);
    
    // 1. Test Q6_K matvec for wqkv (attn_qkv.weight, row 0)
    auto wqkv = open_gguf_tensor(gguf_path, "blk.0.attn_qkv.weight", 210, 256);
    if (!wqkv.fp) return 1;
    
    // Read row 0 blocks
    fseek(wqkv.fp, wqkv.data_offset + wqkv.tensor_offset, SEEK_SET);
    BlockQ6K * wqkv_row0 = new BlockQ6K[wqkv.n_blocks];
    fread(wqkv_row0, wqkv.type_size, wqkv.n_blocks, wqkv.fp);
    
    float cpu_wqkv_row0 = q6k_row_dot(wqkv_row0, wqkv.n_blocks, h2);
    printf("CPU wqkv row 0: %.8f (expect GPU: -0.98120427)\n", cpu_wqkv_row0);
    
    // 2. Test Q5_K matvec for wqkv_gate (attn_gate.weight, row 0)
    auto wqkv_g = open_gguf_tensor(gguf_path, "blk.0.attn_gate.weight", 176, 256);
    if (!wqkv_g.fp) return 1;
    
    fseek(wqkv_g.fp, wqkv_g.data_offset + wqkv_g.tensor_offset, SEEK_SET);
    BlockQ5K * wqkv_g_row0 = new BlockQ5K[wqkv_g.n_blocks];
    fread(wqkv_g_row0, wqkv_g.type_size, wqkv_g.n_blocks, wqkv_g.fp);
    
    float cpu_wqkv_g_row0 = q5k_row_dot(wqkv_g_row0, wqkv_g.n_blocks, h2);
    printf("CPU wqkv_gate row 0: %.8f (expect GPU: 0.39322388)\n", cpu_wqkv_g_row0);
    
    delete[] wqkv_row0;
    delete[] wqkv_g_row0;
    fclose(wqkv.fp);
    fclose(wqkv_g.fp);
    
    // 3. Also compute using ggml_rms_norm for comparison
    // Build a small ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    // Create a tensor for the embedding
    struct ggml_tensor * emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    memcpy(emb->data, h2, 5120 * sizeof(float));
    
    // Read the attn_norm.weight from GGUF
    auto anw = open_gguf_tensor(gguf_path, "blk.0.attn_norm.weight", 4, 1);
    float * anw_data = new float[5120];
    fseek(anw.fp, anw.data_offset + anw.tensor_offset, SEEK_SET);
    fread(anw_data, 4, 5120, anw.fp);
    
    // Create tensor for the weight
    struct ggml_tensor * anw_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    memcpy(anw_t->data, anw_data, 5120 * sizeof(float));
    
    // Compute ggml_rms_norm
    struct ggml_tensor * rms_norm_result = ggml_rms_norm(ctx, emb, 1e-6f);
    struct ggml_tensor * rms_norm_mul = ggml_mul(ctx, rms_norm_result, anw_t);
    
    // Eval
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rms_norm_mul);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    // Compare first few values
    float * ggml_h2 = (float *)rms_norm_mul->data;
    printf("\nRMS Norm comparison (first 8):\n");
    float max_rms_diff = 0;
    for (int i = 0; i < 5120; i++) {
        float diff = fabs(ggml_h2[i] - h2[i]);
        if (diff > max_rms_diff) max_rms_diff = diff;
    }
    for (int i = 0; i < 8; i++) {
        printf("  [%d] ggml=%.8f our=%.8f\n", i, ggml_h2[i], h2[i]);
    }
    printf("Max RMS diff: %.10f\n", max_rms_diff);
    
    ggml_free(ctx);
    delete[] anw_data;
    fclose(anw.fp);
    
    return 0;
}
