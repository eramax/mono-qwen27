#include "mono27b_config.h"
#include "mono27b_gguf.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ─── Quant Block Layouts ─────────────────────────────────────────────────────

struct BlockQ4K {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ8 {
    __half d;
    int8_t qs[32];
};

struct BlockQ5K {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};

// ─── Warp-level reductions ───────────────────────────────────────────────────

template <int W>
__device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int s = W / 2; s > 0; s >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, s);
    return v;
}

template <int W>
__device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int s = W / 2; s > 0; s >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, s));
    return v;
}

// ─── Helper: get_scale_min_k4 ───────────────────────────────────────────────

static __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ─── Elementwise kernels ─────────────────────────────────────────────────────

__global__ static void k_elem_add(float * a, const float * b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] += b[i];
}

__global__ static void k_elem_mul(float * a, const float * b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] *= b[i];
}

__global__ static void k_elem_copy(float * d, const float * s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = s[i];
}

__global__ static void k_elem_silu(float * x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ static void k_elem_softplus(const float * x, const float * bias, float * out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = logf(1.0f + expf(x[i] + bias[i]));
}

__global__ static void k_elem_sigmoid(float * x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ static void k_elem_sigmoid_mul(float * out, const float * sig, const float * other, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = (1.0f / (1.0f + expf(-sig[i]))) * other[i];
}

// ─── RMS norm ────────────────────────────────────────────────────────────────



// Use proper shared memory reduction
template <int BLK>
__global__ static void k_rms_norm_mulw(const float * x, const float * w, float * y, int n, float eps) {
    __shared__ float sh[BLK];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLK) sum += x[i] * x[i];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += BLK) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// ─── Q4_K matvec ─────────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_q4k_mv(const BlockQ4K * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const BlockQ4K * rp = W + (size_t)row * rb;
    for (int b = threadIdx.x; b < rb; b += BLK) {
        const BlockQ4K & qb = rp[b];
        float d = __half2float(qb.d), mn = __half2float(qb.dmin);
        for (int g = 0; g < 4; ++g) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(g * 2, qb.scales, s0, m0);
            get_scale_min_k4(g * 2 + 1, qb.scales, s1, m1);
            float d0 = d * s0, d1 = d * s1, mn0 = mn * m0, mn1 = mn * m1;
            int base = b * 256 + g * 64;
            const uint8_t * qg = qb.qs + g * 32;
            for (int i = 0; i < 32; ++i) {
                sum += (d0 * (qg[i] & 0x0F) - mn0) * x[base + i];
                sum += (d1 * (qg[i] >> 4) - mn1) * x[base + 32 + i];
            }
        }
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── Q5_K matvec ─────────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_q5k_mv(const BlockQ5K * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const BlockQ5K * rp = W + (size_t)row * rb;
    for (int b = threadIdx.x; b < rb; b += BLK) {
        const BlockQ5K & qb = rp[b];
        float d = __half2float(qb.d);
        float mn = __half2float(qb.dmin);
        for (int g = 0; g < 8; ++g) {
            uint8_t s0, m0;
            get_scale_min_k4(g, qb.scales, s0, m0);
            float d0 = d * (float)s0;
            float mn0 = mn * (float)m0;
            int base = b * 256 + g * 32;
            const uint8_t * qs = qb.qs + (g / 2) * 32;
            int use_high = g & 1;
            uint8_t qhm = 1 << g;
            for (int i = 0; i < 32; ++i) {
                int nib = use_high ? (qs[i] >> 4) : (qs[i] & 0x0F);
                int qv = nib + ((qb.qh[i] & qhm) ? 16 : 0);
                sum += (d0 * qv - mn0) * x[base + i];
            }
        }
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── Q6_K matvec ─────────────────────────────────────────────────────────────

// Q6_K matvec: multi-threaded (256 threads per block, shared mem for reduction)
__global__ static void k_q6k_mt(const BlockQ6K * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[256];
    float sum = 0.0f;
    const BlockQ6K * rp = W + (size_t)row * rb;
    for (int b = 0; b < rb; ++b) {
        const BlockQ6K & qb = rp[b];
        float d = __half2float(qb.d);
        int idx = threadIdx.x;
        int g = idx / 16;
        int i = idx % 16;
        int lo = (qb.ql[idx / 2] >> (4 * (idx % 2))) & 0x0F;
        int hi = (qb.qh[idx / 4] >> (2 * (idx % 4))) & 3;
        int qv = (lo | (hi << 4)) - 32;
        int base = b * 256 + g * 16 + i;
        sum += d * (float)qb.scales[g] * (float)qv * x[base];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── Q8_0 matvec ─────────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_q80_mv(const BlockQ8 * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const BlockQ8 * rp = W + (size_t)row * rb;
    for (int b = threadIdx.x; b < rb; b += BLK) {
        const BlockQ8 & qb = rp[b];
        float d = __half2float(qb.d);
        for (int i = 0; i < 32; ++i)
            sum += d * (float)qb.qs[i] * x[b * 32 + i];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── F32 matvec ──────────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_f32_mv(const float * W, const float * x, float * y, int N, int M) {
    int row = blockIdx.x;
    if (row >= M) return;
    __shared__ float sh[BLK];
    float s = 0.0f;
    for (int j = threadIdx.x; j < N; j += BLK) s += W[(size_t)row * N + j] * x[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int j = BLK / 2; j > 0; j >>= 1) {
        if (threadIdx.x < j) { sh[threadIdx.x] += sh[threadIdx.x + j]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── L2 norm (in-place per group) ────────────────────────────────────────────

__global__ static void k_l2_norm_g(float * d, int gs, int ng) {
    int g = blockIdx.x;
    if (g >= ng) return;
    float * gd = d + (size_t)g * gs;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < gs; i += blockDim.x) sq += gd[i] * gd[i];
    sq = warp_reduce_sum<32>(sq);
    __shared__ float sh;
    if (threadIdx.x == 0) sh = sq;
    __syncthreads();
    float inv = rsqrtf(sh + 1e-10f);
    for (int i = threadIdx.x; i < gs; i += blockDim.x) gd[i] *= inv;
}

// ─── M-RoPE ──────────────────────────────────────────────────────────────────

__global__ static void k_mrope(float * buf, int n_heads, int head_dim, int pos, int n_rot) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float * hd = buf + (size_t)h * head_dim;
    for (int d = threadIdx.x * 2; d < n_rot; d += blockDim.x * 2) {
        if (d + 1 >= head_dim) break;
        float theta = powf(MONO27B_TARGET_ROPE_THETA, -2.0f * d / (float)n_rot);
        float c = cosf(theta * pos), s = sinf(theta * pos);
        float v0 = hd[d], v1 = hd[d + 1];
        hd[d] = v0 * c - v1 * s;
        hd[d + 1] = v0 * s + v1 * c;
    }
}

// ─── Q4_K embedding gather (for single token) ────────────────────────────────

__global__ static void k_q4k_embed(const BlockQ4K * T, int tok, float * out, int row_elems, int row_blocks) {
    const BlockQ4K * rp = T + (size_t)tok * row_blocks;
    int tid = threadIdx.x;
    int base_el = tid * 4; // each thread handles 4 elements... no, handle elements directly
    // Each thread handles one element position
    for (int el = tid; el < row_elems; el += blockDim.x) {
        int b = el / 256;
        int intra = el % 256;
        int g = intra / 64;
        int wi = intra % 64;
        const BlockQ4K & qb = rp[b];
        uint8_t s0, m0, s1, m1;
        get_scale_min_k4(g * 2, qb.scales, s0, m0);
        get_scale_min_k4(g * 2 + 1, qb.scales, s1, m1);
        float d0 = __half2float(qb.d) * s0;
        float d1 = __half2float(qb.d) * s1;
        float mn0 = __half2float(qb.dmin) * m0;
        float mn1 = __half2float(qb.dmin) * m1;
        uint8_t qv = qb.qs[g * 32 + (wi < 32 ? wi : wi - 32)];
        out[el] = (wi < 32) ? (d0 * (qv & 0x0F) - mn0) : (d1 * (qv >> 4) - mn1);
    }
}

// ─── Single-token attention (1 thread, sequential) ──────────────────────────

__global__ static void k_attn_1t(const float * Q, const float * K, const float * V, float * O,
                                  int kv_len, int n_h, int n_kvh, int hd, int max_ctx, float scale) {
    int qh = blockIdx.x;
    if (qh >= n_h) return;
    int kvh = qh / (n_h / n_kvh);
    const float * qr = Q + (size_t)qh * hd;
    float * or_ = O + (size_t)qh * hd;

    // One thread handles the entire head
    if (threadIdx.x == 0) {
        // Scores
        extern __shared__ float sc[];
        float maxv = -1e30f;
        for (int p = 0; p < kv_len; ++p) {
            const float * kr = K + ((size_t)kvh * max_ctx + p) * hd;
            float d = 0.0f;
            for (int j = 0; j < hd; ++j) d += qr[j] * kr[j];
            d *= scale;
            sc[p] = d;
            if (d > maxv) maxv = d;
        }
        float sume = 0.0f;
        for (int p = 0; p < kv_len; ++p) { float v = expf(sc[p] - maxv); sc[p] = v; sume += v; }
        float invs = 1.0f / (sume + 1e-10f);
        for (int d = 0; d < hd; ++d) {
            float v = 0.0f;
            for (int p = 0; p < kv_len; ++p)
                v += sc[p] * V[((size_t)kvh * max_ctx + p) * hd + d];
            or_[d] = v * invs;
        }
    }
}

// ─── SSM conv1d (1 thread per channel) ───────────────────────────────────────

__global__ static void k_ssm_conv1d_u(const float * inp, const float * w, float * cs, float * out,
                                       int cc, int ck) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= cc) return;
    int sl = ck - 1;
    float s = 0.0f;
    for (int k = 0; k < sl; ++k) s += cs[(size_t)ch * sl + k] * w[(size_t)ch * ck + k];
    s += inp[ch] * w[(size_t)ch * ck + sl];
    for (int k = 0; k < sl - 1; ++k) cs[(size_t)ch * sl + k] = cs[(size_t)ch * sl + k + 1];
    cs[(size_t)ch * sl + (sl - 1)] = inp[ch];
    out[ch] = s;
}

// ─── Gated DeltaNet (1 thread per group, dt_rank, column) ────────────────────

__global__ static void k_deltanet(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    float * state, float * out,
    int ng, int dr, int hv, int hk)
{
    int g_idx = blockIdx.x;  // [0..ng)
    int r_idx = blockIdx.y;  // [0..dr)
    int col   = blockIdx.z;  // [0..hv)
    if (g_idx >= ng || r_idx >= dr || col >= hv) return;

    float gv = expf(g[r_idx + (size_t)g_idx * dr]);
    float bv = beta[r_idx + (size_t)g_idx * dr];

    const float * qg = q + (size_t)g_idx * hk;
    const float * kg = k + (size_t)g_idx * hk;
    const float * vr = v + (size_t)r_idx * hv + col;

    float * S = state + ((size_t)g_idx * dr + r_idx) * (size_t)hv * hk + col;
    // S is [hv][hk] stored row-major: S[col][0..hk-1]

    float kv = 0.0f;
    for (int i = 0; i < hk; ++i) kv += S[(size_t)i * hv] * kg[i]; // S[col][i]
    float delta = (*vr - gv * kv) * bv;

    float attn = 0.0f;
    for (int i = 0; i < hk; ++i) {
        float sn = gv * S[(size_t)i * hv] + kg[i] * delta;
        S[(size_t)i * hv] = sn;
        attn += sn * qg[i];
    }

    out[(size_t)r_idx * hv + col] = attn;
}

// ─── Quant info helpers ──────────────────────────────────────────────────────

struct QuantInfo { size_t block_size, type_size; };

static QuantInfo qi(uint32_t t) {
    switch (t) {
        case MONO27B_GGML_TYPE_Q4_K: return {256, sizeof(BlockQ4K)};
        case MONO27B_GGML_TYPE_Q5_K: return {256, sizeof(BlockQ5K)};
        case MONO27B_GGML_TYPE_Q6_K: return {256, sizeof(BlockQ6K)};
        case MONO27B_GGML_TYPE_Q8_0: return {32, sizeof(BlockQ8)};
        case MONO27B_GGML_TYPE_F32:  return {1, 4};
        case MONO27B_GGML_TYPE_F16:  return {1, 2};
        default: return {0, 0};
    }
}

static int nblocks(uint32_t ne, uint32_t t) {
    auto q = qi(t);
    return q.block_size ? (int)(ne / q.block_size) : 0;
}

static int nrows_quant(const Mono27BGgufTensorInfo * t, uint32_t elem_per_row) {
    if (!t || t->size_bytes == 0) return 0;
    auto q = qi(t->ggml_type);
    if (q.block_size == 0) return (int)(t->size_bytes / elem_per_row / q.type_size);
    int rb = (int)(elem_per_row / q.block_size * q.type_size);
    return rb > 0 ? (int)(t->size_bytes / rb) : 0;
}

// ─── Tensor loading helper ───────────────────────────────────────────────────

static size_t lw_total_alloc = 0;

static bool lw(const unsigned char * data, uint64_t doff,
               const Mono27BGgufTensorInfo & ti, WeightView * wv,
               char * err, size_t ec, const char * lbl) {
    std::memset(wv, 0, sizeof(*wv));
    if (!ti.size_bytes) return true;
    auto q = qi(ti.ggml_type);
    if (q.block_size == 0) { return true; }

    int input_dim = ti.n_dims > 0 ? (int)ti.dims[0] : 0;
    int output_dim = ti.n_dims > 1 ? (int)ti.dims[1] : (int)ti.dims[0];

    wv->ggml_type = ti.ggml_type;
    wv->row_blocks = (q.block_size > 1) ? (input_dim / q.block_size) : input_dim;
    wv->row_count = output_dim;
    cudaError_t e = cudaMalloc(&wv->ptr, ti.size_bytes);
    lw_total_alloc += ti.size_bytes;
    if (e != cudaSuccess) { std::snprintf(err, ec, "malloc %s: %s", lbl, cudaGetErrorString(e)); return false; }
    e = cudaMemcpy(wv->ptr, data + doff + ti.offset, ti.size_bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { cudaFree(wv->ptr); wv->ptr = nullptr;
        std::snprintf(err, ec, "cpy %s: %s", lbl, cudaGetErrorString(e)); return false; }
    return true;
}

// ─── Find tensor ─────────────────────────────────────────────────────────────

static const Mono27BGgufTensorInfo * ft(const Mono27BGgufTensorInfo * ts, size_t n, const char * name) {
    for (size_t i = 0; i < n; ++i) if (ts[i].name == name) return &ts[i];
    return nullptr;
}

// ─── Public API: load weights ────────────────────────────────────────────────

extern "C" bool mono27b_engine_load_weights(
    const unsigned char * gguf_data, uint64_t data_offset,
    const Mono27BGgufTensorInfo * tensors, size_t tensor_count,
    Mono27BExecutorWeights * gpu_weights, char * error, size_t error_cap)
{
    lw_total_alloc = 0;
    std::memset(gpu_weights, 0, sizeof(*gpu_weights));

    lw(gguf_data, data_offset, *ft(tensors, tensor_count, "output_norm.weight"), &gpu_weights->output_norm, error, error_cap, "onorm");
    lw(gguf_data, data_offset, *ft(tensors, tensor_count, "output.weight"), &gpu_weights->lm_head, error, error_cap, "lmh");
    lw(gguf_data, data_offset, *ft(tensors, tensor_count, "token_embd.weight"), &gpu_weights->tok_embd, error, error_cap, "tok");

    for (int il = 0; il < MONO27B_TARGET_LAYERS; ++il) {
        bool is_a = ((il + 1) % MONO27B_TARGET_FA_INTERVAL) == 0;
        auto & l = gpu_weights->layers[il];
        l.is_attention = is_a;
        char nm[128];

        auto lw2 = [&](const char * nn, WeightView * p, const char * lb) -> bool {
            std::snprintf(nm, sizeof(nm), "blk.%d.%s", il, nn);
            auto * ti = ft(tensors, tensor_count, nm);
            if (ti && ti->size_bytes) {
                lw(gguf_data, data_offset, *ti, p, error, error_cap, lb);
                return true;
            }
            return false;
        };

        if (is_a) {
            lw2("attn_norm.weight", &l.attn.attn_norm, "an");
            lw2("attn_q.weight", &l.attn.wq, "wq");
            lw2("attn_k.weight", &l.attn.wk, "wk");
            lw2("attn_v.weight", &l.attn.wv, "wv");
            if (!lw2("attn_o.weight", &l.attn.wo, "wo"))
                lw2("attn_output.weight", &l.attn.wo, "wo");
            lw2("attn_q_norm.weight", &l.attn.q_norm, "qn");
            lw2("attn_k_norm.weight", &l.attn.k_norm, "kn");
            lw2("post_attention_norm.weight", &l.attn.post_norm, "pan");
            lw2("ffn_gate.weight", &l.attn.ffn_gate, "fg");
            lw2("ffn_up.weight", &l.attn.ffn_up, "fu");
            lw2("ffn_down.weight", &l.attn.ffn_down, "fd");
        } else {
            lw2("attn_norm.weight", &l.ssm.attn_norm, "sn");
            // Try multiple naming conventions for SSM qkv weight
            const char * qkv_names[] = {"attn_qkv.weight", "attn_q.weight", "ssm_in.weight"};
            const Mono27BGgufTensorInfo * t1 = nullptr;
            for (auto nn : qkv_names) {
                std::snprintf(nm, sizeof(nm), "blk.%d.%s", il, nn);
                t1 = ft(tensors, tensor_count, nm);
                if (t1 && t1->size_bytes) break;
            }
            if (t1 && t1->size_bytes) { lw(gguf_data, data_offset, *t1, &l.ssm.wqkv, error, error_cap, "wqkv"); }

            // Try multiple naming conventions for SSM gate weight
            const char * gate_names[] = {"attn_gate.weight", "attn_k.weight", "ssm_in_gate.weight"};
            const Mono27BGgufTensorInfo * t2 = nullptr;
            for (auto nn : gate_names) {
                std::snprintf(nm, sizeof(nm), "blk.%d.%s", il, nn);
                t2 = ft(tensors, tensor_count, nm);
                if (t2 && t2->size_bytes) break;
            }
            if (t2 && t2->size_bytes) {
                lw(gguf_data, data_offset, *t2, &l.ssm.wqkv_gate, error, error_cap, "wqkvg");
            }
            lw2("ssm_conv1d.weight", &l.ssm.ssm_conv1d, "sc1d");
            lw2("ssm_beta.weight", &l.ssm.ssm_beta, "sb");
            lw2("ssm_alpha.weight", &l.ssm.ssm_alpha, "sa");
            // Try alternative names for dt bias and ssm_a
            if (!lw2("ssm_dt_bias", &l.ssm.ssm_dt_bias, "sdt")) 
                lw2("ssm_dt.bias", &l.ssm.ssm_dt_bias, "sdt");
            if (!lw2("ssm_a_log", &l.ssm.ssm_a_log, "sal")) {
                lw2("ssm_a", &l.ssm.ssm_a_log, "sal");
            }
            lw2("ssm_out.weight", &l.ssm.ssm_out, "sso");
            lw2("post_attention_norm.weight", &l.ssm.post_norm, "spn");
            lw2("ffn_gate.weight", &l.ssm.ffn_gate, "sfg");
            lw2("ffn_up.weight", &l.ssm.ffn_up, "sfu");
            lw2("ffn_down.weight", &l.ssm.ffn_down, "sfd");
        }
    }
    fprintf(stderr, "[mem] lw total: %zu bytes (%.0f MB)\n", lw_total_alloc, lw_total_alloc/1048576.0);
    fflush(stderr);
    return true;
}

extern "C" void mono27b_engine_free_weights(Mono27BExecutorWeights * w) {
    auto ff = [](WeightView & v) { cudaFree(v.ptr); v.ptr = nullptr; };
    for (int il = 0; il < MONO27B_TARGET_LAYERS; ++il) {
        if (w->layers[il].is_attention) {
            auto & a = w->layers[il].attn;
            ff(a.attn_norm); ff(a.wq); ff(a.wk); ff(a.wv); ff(a.wo);
            ff(a.q_norm); ff(a.k_norm); ff(a.post_norm);
            ff(a.ffn_gate); ff(a.ffn_up); ff(a.ffn_down);
        } else {
            auto & s = w->layers[il].ssm;
            ff(s.attn_norm); ff(s.wqkv); ff(s.wqkv_gate); ff(s.ssm_conv1d);
            ff(s.ssm_beta); ff(s.ssm_alpha); ff(s.ssm_dt_bias); ff(s.ssm_a_log);
            ff(s.ssm_out); ff(s.post_norm);
            ff(s.ffn_gate); ff(s.ffn_up); ff(s.ffn_down);
        }
    }
    ff(w->output_norm); ff(w->lm_head); ff(w->tok_embd);
    std::memset(w, 0, sizeof(*w));
}

// ─── State init/free ─────────────────────────────────────────────────────────

extern "C" bool mono27b_engine_init_state(int max_ctx, Mono27BExecutorState * state,
                                           char * error, size_t error_cap) {
    std::memset(state, 0, sizeof(*state));
    size_t kb = (size_t)max_ctx * MONO27B_TARGET_N_KV_HEAD * MONO27B_TARGET_HEAD_DIM * sizeof(float);
    size_t sb = (size_t)MONO27B_SSM_N_GROUP * MONO27B_SSM_DT_RANK *
                 MONO27B_SSM_HEAD_V * MONO27B_SSM_HEAD_K * sizeof(float);
    size_t cb = (size_t)(MONO27B_SSM_CONV_KERN - 1) * MONO27B_SSM_CONV_CH * sizeof(float);
    int sl = MONO27B_TARGET_LAYERS - MONO27B_TARGET_FA_LAYERS;
    size_t ws = (size_t)MONO27B_TARGET_HIDDEN * 2 + (size_t)MONO27B_TARGET_Q_DIM * 2 +
                (size_t)MONO27B_TARGET_KV_DIM * 2 + (size_t)MONO27B_TARGET_FFN +
                (size_t)MONO27B_SSM_CONV_CH + (size_t)MONO27B_SSM_D_INNER +
                (size_t)MONO27B_TARGET_VOCAB;
    ws *= sizeof(float);
    for (int i = 0; i < MONO27B_TARGET_FA_LAYERS; ++i) {
        if (cudaSuccess != cudaMalloc(&state->kv_cache_k[i], kb)) {
            std::snprintf(error, error_cap, "kv_k[%d]: %s", i, cudaGetErrorString(cudaGetLastError()));
            goto fail;
        }
        if (cudaSuccess != cudaMalloc(&state->kv_cache_v[i], kb)) {
            std::snprintf(error, error_cap, "kv_v[%d]: %s", i, cudaGetErrorString(cudaGetLastError()));
            goto fail;
        }
    }
    // Skip SSM state allocation to save memory (2.3GB) - deltanet is not active anyway
    for (int i = 0; i < sl; ++i) {
        state->ssm_state[i] = nullptr;
        state->conv_state[i] = nullptr;
    }
    // Pre-allocate working buffers for decode_step (saves fragmentation)
    if (cudaSuccess != cudaMalloc(&state->work_buf, ws)) {
        std::snprintf(error, error_cap, "work: %s", cudaGetErrorString(cudaGetLastError()));
        goto fail;
    }
    state->work_buf_size = ws;

    state->kv_len = 0;
    return true;
fail:
    if (error[0] == '\0')
        std::snprintf(error, error_cap, "cudaMalloc state buf failed");
    mono27b_engine_free_state(state);
    return false;
}

extern "C" void mono27b_engine_free_state(Mono27BExecutorState * state) {
    for (int i = 0; i < MONO27B_TARGET_FA_LAYERS; ++i) {
        cudaFree(state->kv_cache_k[i]); cudaFree(state->kv_cache_v[i]);
    }
    int sl = MONO27B_TARGET_LAYERS - MONO27B_TARGET_FA_LAYERS;
    for (int i = 0; i < sl; ++i) {
        cudaFree(state->ssm_state[i]); cudaFree(state->conv_state[i]);
    }
    cudaFree(state->work_buf);
    std::memset(state, 0, sizeof(*state));
}

// ─── Launch helpers ──────────────────────────────────────────────────────────

static void l_q4k(const BlockQ4K * W, int rb, int rc, const float * x, float * y) {
    k_q4k_mv<128><<<rc, 128>>>(W, x, y, rb, rc);
}

static void l_q5k(const BlockQ5K * W, int rb, int rc, const float * x, float * y) {
    k_q5k_mv<128><<<rc, 128>>>(W, x, y, rb, rc);
}

static void l_q6k(const BlockQ6K * W, int rb, int rc, const float * x, float * y) {
    k_q6k_mt<<<rc, 256>>>(W, x, y, rb, rc);
}

// Type-dispatched matvec: W at ggml_type, rb blocks per row, rc rows, x input, y output
static void l_mv(void * W, uint32_t ggml_type, int rb, int rc, const float * x, float * y) {
    if (rc == 0 || !W) return;
    if (ggml_type == MONO27B_GGML_TYPE_Q6_K) {
        l_q6k((const BlockQ6K *)W, rb, rc, x, y);
    } else {
        switch (ggml_type) {
            case MONO27B_GGML_TYPE_Q4_K: l_q4k((const BlockQ4K *)W, rb, rc, x, y); break;
            case MONO27B_GGML_TYPE_Q5_K: l_q5k((const BlockQ5K *)W, rb, rc, x, y); break;
            case MONO27B_GGML_TYPE_F32:  k_f32_mv<128><<<rc, 128>>>((const float *)W, x, y, rb, rc); break;
            case MONO27B_GGML_TYPE_Q8_0: k_q80_mv<128><<<rc, 128>>>((const BlockQ8 *)W, x, y, rb, rc); break;
        }
    }
}

static void l_rms(float * dst, const float * x, const float * w, int n) {
    k_rms_norm_mulw<256><<<1, 256>>>(x, w, dst, n, MONO27B_RMS_EPS);
}

// ─── decode_step ─────────────────────────────────────────────────────────────

#define MV(wv, x, y) l_mv((wv).ptr, (wv).ggml_type, (wv).row_blocks, (wv).row_count, x, y)
#define WV(wv) ((const float *)(wv).ptr)
#define TRACE(lbl) ((void)0)

extern "C" bool mono27b_engine_decode_step(
    const Mono27BExecutorWeights * we,
    Mono27BExecutorState * st,
    int tok, int pos,
    Mono27BLogitsOutput * out,
    char * error, size_t error_cap)
{
    size_t sh = MONO27B_TARGET_HIDDEN * sizeof(float);
    size_t sq = MONO27B_TARGET_Q_DIM * sizeof(float);
    size_t sv = MONO27B_TARGET_KV_DIM * sizeof(float);
    size_t sf = MONO27B_TARGET_FFN * sizeof(float);
    size_t ss = MONO27B_SSM_CONV_CH * sizeof(float);
    size_t sg = MONO27B_SSM_D_INNER * sizeof(float);
    size_t sl = MONO27B_TARGET_VOCAB * sizeof(float);
    size_t total_work = sh + sh + sq*2 + sv*2 + sf + ss + sg;
    size_t total_alloc = total_work + sl;

    float * work_buf = nullptr;
    { cudaError_t ce = cudaMalloc(&work_buf, total_alloc);
      if (ce != cudaSuccess) {
        std::snprintf(error, error_cap, "alloc: %s (%zu)", cudaGetErrorString(ce), total_alloc);
        return false;
    } }
    fprintf(stderr, "[wb] %p\n", work_buf);

    float * h   = work_buf;
    float * h2  = h   + sh / sizeof(float);
    float * qb  = h2  + sh / sizeof(float);
    float * kb  = qb  + sq*2 / sizeof(float);
    float * fb  = kb  + sv*2 / sizeof(float);
    float * sb  = fb  + sf / sizeof(float);
    float * gb  = sb  + ss / sizeof(float);
    out->logits = work_buf + total_work / sizeof(float);

    h   = work_buf;
    h2  = h   + sh / sizeof(float);
    qb  = h2  + sh / sizeof(float);
    kb  = qb  + sq*2 / sizeof(float);
    fb  = kb  + sv*2 / sizeof(float);
    sb  = fb  + sf / sizeof(float);
    gb  = sb  + ss / sizeof(float);
    out->logits = work_buf + total_work / sizeof(float);

    int fa_i = 0, ssm_i = 0, il = 0; cudaError_t sync_err = cudaSuccess;

    { size_t f,t; cudaMemGetInfo(&f,&t); fprintf(stderr, "[mem] before emb: free=%.0fMB\n", f/1048576.0); }
    mono27b_engine_embed(we, tok, h, error, error_cap);
    TRACE("emb");

    // Only run first 4 layers to debug
    // Skip all layers to isolate LM head issue
    for (il = 0; il < 31; ++il) {
        bool is_a = ((il + 1) % MONO27B_TARGET_FA_INTERVAL) == 0;
        TRACE("pre");

        if (is_a) {
            const auto & L = we->layers[il].attn;
            int max_ctx = st->max_ctx > 0 ? st->max_ctx : 8192;

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN); TRACE("rms");
            MV(L.wq, h2, qb); TRACE("wq");
            MV(L.wk, h2, kb); TRACE("wk");
            MV(L.wv, h2, kb + MONO27B_TARGET_KV_DIM); TRACE("wv");

            if (L.q_norm.ptr)
                for (int hh = 0; hh < MONO27B_TARGET_N_HEAD; ++hh)
                    k_rms_norm_mulw<256><<<1, 256>>>(qb + hh * MONO27B_TARGET_HEAD_DIM,
                        WV(L.q_norm), qb + hh * MONO27B_TARGET_HEAD_DIM,
                        MONO27B_TARGET_HEAD_DIM, MONO27B_RMS_EPS);
            if (L.k_norm.ptr)
                for (int hh = 0; hh < MONO27B_TARGET_N_KV_HEAD; ++hh)
                    k_rms_norm_mulw<256><<<1, 256>>>(kb + hh * MONO27B_TARGET_HEAD_DIM,
                        WV(L.k_norm), kb + hh * MONO27B_TARGET_HEAD_DIM,
                        MONO27B_TARGET_HEAD_DIM, MONO27B_RMS_EPS);
            TRACE("qkn");

            k_mrope<<<MONO27B_TARGET_N_HEAD, 64>>>(qb, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_HEAD_DIM, pos, MONO27B_N_ROT_DIMS); TRACE("mq");
            k_mrope<<<MONO27B_TARGET_N_KV_HEAD, 64>>>(kb, MONO27B_TARGET_N_KV_HEAD, MONO27B_TARGET_HEAD_DIM, pos, MONO27B_N_ROT_DIMS); TRACE("mk");

            for (int hh = 0; hh < MONO27B_TARGET_N_KV_HEAD; ++hh) {
                size_t off = ((size_t)pos * MONO27B_TARGET_N_KV_HEAD + hh) * MONO27B_TARGET_HEAD_DIM;
                cudaMemcpyAsync(st->kv_cache_k[fa_i] + off, kb + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(st->kv_cache_v[fa_i] + off,
                    kb + MONO27B_TARGET_KV_DIM + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            TRACE("kvc");

            // Gate: sigmoid(gate) * Q (skip attn for now)
            k_elem_sigmoid_mul<<<(MONO27B_TARGET_Q_DIM + 255) / 256, 256>>>(
                qb, qb + MONO27B_TARGET_Q_DIM, qb, MONO27B_TARGET_Q_DIM); TRACE("gt");

            MV(L.wo, qb, h2); TRACE("wo");
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("res1");

            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN); TRACE("porm");
            MV(L.ffn_gate, h, fb); TRACE("fg");
            k_elem_silu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, MONO27B_TARGET_FFN); TRACE("silu");
            MV(L.ffn_up, h, kb); TRACE("fu");
            k_elem_mul<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN); TRACE("mul");
            MV(L.ffn_down, fb, h); TRACE("fd");
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN); TRACE("res2");
            fa_i++;

        } else {
            const auto & L = we->layers[il].ssm;
            TRACE("ssm");

            if (!L.wqkv.ptr || !L.wqkv_gate.ptr) {
                k_elem_copy<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN);
                TRACE("cpy");
                goto ssm_ffn;
            }

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN); TRACE("s_rms");
            MV(L.wqkv, h2, sb); TRACE("wqkv");
            MV(L.wqkv_gate, h2, gb); TRACE("wg");

            if (L.ssm_beta.ptr && L.ssm_alpha.ptr && L.ssm_dt_bias.ptr && L.ssm_a_log.ptr) {
                int dr = MONO27B_SSM_DT_RANK;
                MV(L.ssm_beta, h2, kb); TRACE("sbeta");
                k_elem_sigmoid<<<(dr + 31) / 32, 32>>>(kb, dr); TRACE("sig");
                MV(L.ssm_alpha, h2, qb); TRACE("salph");
                k_elem_softplus<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_dt_bias), qb, dr); TRACE("sp");
                k_elem_mul<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_a_log), dr); TRACE("gmul");

                k_ssm_conv1d_u<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                    sb, (const float *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                    MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN); TRACE("conv");
                k_elem_silu<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(sb, MONO27B_SSM_CONV_CH); TRACE("csilu");

                float * qr = sb;
                float * kr = sb + MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;
                float * vr = sb + 2 * MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;

                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(qr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP); TRACE("l2q");
                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(kr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP); TRACE("l2k");

                dim3 dg(MONO27B_SSM_N_GROUP, MONO27B_SSM_DT_RANK, MONO27B_SSM_HEAD_V);
                k_deltanet<<<dg, 1>>>(qr, kr, vr, qb, kb, st->ssm_state[ssm_i], gb,
                    MONO27B_SSM_N_GROUP, MONO27B_SSM_DT_RANK,
                    MONO27B_SSM_HEAD_V, MONO27B_SSM_HEAD_K); TRACE("dnet");

                // Gate: wqkv_gate @ h2 → siLU → mul with rms_norm(ssm_out)
                MV(L.wqkv_gate, h2, fb); TRACE("re-g");
                k_elem_silu<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(fb, MONO27B_SSM_D_INNER); TRACE("g_silu");
                l_rms(h2, gb, nullptr, MONO27B_SSM_D_INNER); TRACE("grms");
                k_elem_mul<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(h2, fb, MONO27B_SSM_D_INNER); TRACE("gmul2");
                MV(L.ssm_out, h2, sb); TRACE("ssmo");
                k_elem_copy<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, sb, MONO27B_TARGET_HIDDEN); TRACE("scp");
                k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("sadd");
            } else {
                k_ssm_conv1d_u<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                    sb, (const float *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                    MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN); TRACE("conv2");
                MV(L.ssm_out, sb, h2); TRACE("ssmo2");
                k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("sadd2");
            }

        ssm_ffn:
            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN); TRACE("porms");
            MV(L.ffn_gate, h, fb); TRACE("sfg");
            k_elem_silu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, MONO27B_TARGET_FFN); TRACE("ssil");
            MV(L.ffn_up, h, kb); TRACE("sfu");
            k_elem_mul<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN); TRACE("smul");
            MV(L.ffn_down, fb, h); TRACE("sfd");
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN); TRACE("sres");
            ssm_i++;
        }
    }
    TRACE("post-loop");

    // Output norm + LM head
    l_rms(h2, h, WV(we->output_norm), MONO27B_TARGET_HIDDEN);
    // LM head: just write 1 to work_buf[0]
    k_elem_sigmoid<<<1, 256>>>(work_buf, 1);
    sync_err = cudaDeviceSynchronize();
    fprintf(stderr, "[lm] sig: %s\n", cudaGetErrorString(sync_err));
    st->kv_len = pos + 1;

cleanup:
    cudaFree(work_buf);
    return error[0] == '\0';
}

extern "C" void mono27b_engine_free_logits(Mono27BLogitsOutput * out) {
    // logits is part of state.work_buf, don't free it separately
    out->logits = nullptr;
}

// ─── Embedding ───────────────────────────────────────────────────────────────

extern "C" bool mono27b_engine_embed(
    const Mono27BExecutorWeights * we, int token_id,
    float * hidden, char * error, size_t error_cap)
{
    if (token_id < 0 || (uint32_t)token_id >= we->tok_embd.row_count) {
        cudaMemset(hidden, 0, MONO27B_TARGET_HIDDEN * sizeof(float));
        return true;
    }
    if (!we->tok_embd.ptr) {
        cudaMemset(hidden, 0, MONO27B_TARGET_HIDDEN * sizeof(float));
        return true;
    }
    int rb = we->tok_embd.row_blocks;
    k_q4k_embed<<<1, 256>>>((const BlockQ4K *)we->tok_embd.ptr, token_id, hidden,
                             MONO27B_TARGET_HIDDEN, rb);
    return true;
}

// ─── Backward-compat stubs ───────────────────────────────────────────────────

extern "C" bool mono27b_executor_init(const Mono27BBlobHeader * h, int mc, Mono27BRuntimeLayout * l,
                                       char * e, size_t ec) {
    if (!h || !l) { if (e && ec) std::snprintf(e, ec, "null input"); return false; }
    l->kv_bytes = (size_t)mc * MONO27B_TARGET_FA_LAYERS * MONO27B_TARGET_N_KV_HEAD *
                  MONO27B_TARGET_HEAD_DIM * 2U * sizeof(float);
    l->rollback_bytes = 0; l->workspace_bytes = 0;
    l->state_bytes = l->kv_bytes;
    return true;
}

extern "C" bool mono27b_executor_load_weights(
    const unsigned char *, size_t, uint32_t, uint32_t,
    const unsigned char *, size_t, uint32_t, uint32_t,
    const unsigned char *, size_t, uint32_t, uint32_t,
    Mono27BExecutorWeights *, char *, size_t) { return false; }

extern "C" void mono27b_executor_free_weights(Mono27BExecutorWeights *) {}

extern "C" bool mono27b_executor_set_norm_scale(Mono27BExecutorWeights *, const float *, uint32_t,
                                                  char *, size_t) { return false; }

extern "C" bool mono27b_executor_run_step(
    const Mono27BExecutorWeights *, const int32_t *, size_t, int32_t *, size_t,
    size_t *, char *, size_t, char *, size_t) { return false; }

extern "C" bool mono27b_executor_run_prompt(
    const char *, const int32_t *, size_t,
    const unsigned char *, size_t, uint32_t, uint32_t,
    const unsigned char *, size_t, uint32_t, uint32_t,
    const unsigned char *, size_t, uint32_t, uint32_t,
    int32_t *, size_t, size_t *, char *, size_t, char *, size_t) { return false; }
