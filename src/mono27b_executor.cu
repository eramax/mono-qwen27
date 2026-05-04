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

struct BlockQ8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

struct BlockIQ4XS {
    __half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};

// kvalues for IQ4_XS dequantization
static constexpr __device__ float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
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

static __host__ __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static __host__ __device__ inline float ggml_fp16_to_fp32(uint16_t h) {
    union U32F { uint32_t u; float f; };
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    U32F normalized_bits{(two_w >> 4) + exp_offset};
    U32F normalized_value{0};
    normalized_value.f = normalized_bits.f * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    U32F denormalized_bits{(two_w >> 17) | magic_mask};
    U32F denormalized_value{0};
    denormalized_value.f = denormalized_bits.f - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    U32F result_bits{sign |
        (two_w < denormalized_cutoff ? denormalized_value.u : normalized_value.u)};
    return result_bits.f;
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float z = x[i] + bias[i];
        out[i] = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
    }
}

__global__ static void k_elem_sigmoid(float * x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ static void k_elem_sigmoid_mul(float * out, const float * sig, const float * other, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = (1.0f / (1.0f + expf(-sig[i]))) * other[i];
}

__global__ static void k_elem_swiglu(float * out, const float * up, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float x = up[i];
        out[i] = out[i] * (x / (1.0f + expf(-x)));
    }
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

// ─── F16 matvec ─────────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_f16_mv(const __half * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const __half * rp = W + (size_t)row * rb;
    for (int j = threadIdx.x; j < rb; j += BLK) {
        sum += __half2float(rp[j]) * x[j];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
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
        float d = ggml_fp16_to_fp32(__half_as_ushort(qb.d));
        float mn = ggml_fp16_to_fp32(__half_as_ushort(qb.dmin));
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

static __device__ inline float q5k_val(const BlockQ5K & qb, int e) {
    const int n64 = e / 64;
    const int l   = e % 32;
    const int hi  = (e % 64) / 32;
    uint8_t sc, m;
    get_scale_min_k4(n64 * 2 + hi, qb.scales, sc, m);
    const float d = __half2float(qb.d);
    const float mn = __half2float(qb.dmin);
    const float d0 = d * (float)sc;
    const float m0 = mn * (float)m;
    const uint8_t qsrc = qb.qs[n64 * 32 + l];
    const int qv = (hi == 0 ? (qsrc & 0x0F) : (qsrc >> 4)) +
                   ((qb.qh[l] & (1 << (n64 * 2 + hi))) ? 16 : 0);
    return d0 * (float)qv - m0;
}

template <int BLK>
__global__ static void k_q5k_mv(const BlockQ5K * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const BlockQ5K * rp = W + (size_t)row * rb;
    for (int b = 0; b < rb; ++b) {
        const BlockQ5K & qb = rp[b];
        for (int e = threadIdx.x; e < 256; e += BLK)
            sum += q5k_val(qb, e) * x[b * 256 + e];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── IQ4_XS matvec ───────────────────────────────────────────────────────────

template <int BLK>
__global__ static void k_iq4xs_mv(const BlockIQ4XS * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[BLK];
    float sum = 0.0f;
    const BlockIQ4XS * rp = W + (size_t)row * rb;
    for (int b = threadIdx.x; b < rb; b += BLK) {
        const BlockIQ4XS & qb = rp[b];
        float d = __half2float(qb.d);
        for (int g = 0; g < 8; ++g) {
            int ls = (qb.scales_l[g/2] >> (4*(g%2)) & 0x0F) | (((qb.scales_h >> (2*g)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            int base = b * 256 + g * 32;
            const uint8_t * qg = qb.qs + g * 16;
            for (int i = threadIdx.x; i < 16; i += BLK) {
                sum += dl * kvalues_iq4nl[qg[i] & 0x0F] * x[base + i];
                sum += dl * kvalues_iq4nl[qg[i] >> 4] * x[base + 16 + i];
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

static __host__ __device__ inline float q6k_val(const BlockQ6K & qb, int e) {
    const int n128 = e / 128;
    const int l    = e % 32;
    const int r    = (e % 128) / 32;
    const int is   = l / 16;
    const uint8_t * ql = qb.ql + n128 * 64;
    const uint8_t * qh = qb.qh + n128 * 32;
    const int8_t  * sc = qb.scales + n128 * 8;

    int qv;
    if (r == 0) {
        qv = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
    } else if (r == 1) {
        qv = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
    } else if (r == 2) {
        qv = (int)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    } else {
        qv = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    }

    const int sc_idx = is + (r * 2);
    return __half2float(qb.d) * (float) sc[sc_idx] * (float) qv;
}

static float q6k_row_dot_cpu(const BlockQ6K * row, int rb, const float * x) {
    float sum = 0.0f;
    for (int b = 0; b < rb; ++b) {
        const BlockQ6K & qb = row[b];
        for (int e = 0; e < 256; ++e) {
            sum += q6k_val(qb, e) * x[b * 256 + e];
        }
    }
    return sum;
}

static void debug_probe_q6k_rows(FILE * fp, const char * phase, int step, int pos, int tok,
                                 const BlockQ6K * gpu_rows, int rb, int rc,
                                 const float * hidden_dev, const float * gpu_logits_dev,
                                 int probe_rows = 8) {
    if (!fp || !gpu_rows || !hidden_dev || !gpu_logits_dev || rb <= 0 || rc <= 0) return;

    const int rows = std::min(probe_rows, rc);
    if (rows <= 0) return;

    std::vector<float> hidden(MONO27B_TARGET_HIDDEN);
    std::vector<float> gpu_logits(rows);
    std::vector<BlockQ6K> row_buf(static_cast<size_t>(rows) * static_cast<size_t>(rb));

    cudaError_t e = cudaMemcpy(hidden.data(), hidden_dev,
                               MONO27B_TARGET_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(fp, "%s\t%d\t%d\t%d\tlm_head_q6k_probe\tcopy_hidden\t%s\n",
                     phase, step, pos, tok, cudaGetErrorString(e));
        return;
    }

    e = cudaMemcpy(gpu_logits.data(), gpu_logits_dev,
                   static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(fp, "%s\t%d\t%d\t%d\tlm_head_q6k_probe\tcopy_logits\t%s\n",
                     phase, step, pos, tok, cudaGetErrorString(e));
        return;
    }

    e = cudaMemcpy(row_buf.data(), gpu_rows,
                   static_cast<size_t>(rows) * static_cast<size_t>(rb) * sizeof(BlockQ6K),
                   cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(fp, "%s\t%d\t%d\t%d\tlm_head_q6k_probe\tcopy_rows\t%s\n",
                     phase, step, pos, tok, cudaGetErrorString(e));
        return;
    }

    double max_abs = 0.0;
    int worst_row = 0;
    for (int row = 0; row < rows; ++row) {
        const float cpu = q6k_row_dot_cpu(row_buf.data() + static_cast<size_t>(row) * rb, rb, hidden.data());
        const float gpu = gpu_logits[row];
        const float delta = cpu - gpu;
        const double abs_delta = std::fabs((double)delta);
        if (abs_delta > max_abs) {
            max_abs = abs_delta;
            worst_row = row;
        }
        std::fprintf(fp, "%s\t%d\t%d\t%d\tlm_head_q6k_row\t%d\t%.9g\t%.9g\t%.9g\n",
                     phase, step, pos, tok, row, gpu, cpu, delta);
    }

    std::fprintf(fp, "%s\t%d\t%d\t%d\tlm_head_q6k_summary\trows=%d\tworst_row=%d\tmax_abs=%.9g\n",
                 phase, step, pos, tok, rows, worst_row, max_abs);
}

// Q6_K matvec: direct dequantization matching ggml reference
__global__ static void k_q6k_mt(const BlockQ6K * W, const float * x, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[256];
    float sum = 0.0f;
    const BlockQ6K * rp = W + (size_t)row * rb;
    for (int b = 0; b < rb; ++b) {
        const BlockQ6K & qb = rp[b];
        for (int e = threadIdx.x; e < 256; e += 256) {
            sum += q6k_val(qb, e) * x[b * 256 + e];
        }
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
    __shared__ float sh[128];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < gs; i += blockDim.x) sq += gd[i] * gd[i];
    sh[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float sum_sq = sh[0];
    float inv = rsqrtf(fmaxf(sum_sq, MONO27B_RMS_EPS * MONO27B_RMS_EPS));
    for (int i = threadIdx.x; i < gs; i += blockDim.x) gd[i] *= inv;
}

// ─── M-RoPE ──────────────────────────────────────────────────────────────────
// Qwen3.5 uses multi-section RoPE with sections [t, h, w, _] = [11, 11, 10, 0].
// For text-only inference, only section 0 (text, 11 pairs = 22 dims) is rotated
// with the token position; sections 1 and 2 use position 0 (no rotation).
// total n_rot is the sum of all sections: 11+11+10+0 = 32 pairs = 64 dims.

__global__ static void k_mrope(float * buf, int n_heads, int head_dim, int pos, int n_rot_section0) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float * hd = buf + (size_t)h * head_dim;
    for (int d = threadIdx.x * 2; d < n_rot_section0; d += blockDim.x * 2) {
        if (d + 1 >= head_dim) break;
        float theta = powf(MONO27B_TARGET_ROPE_THETA, -2.0f * d / (float)n_rot_section0);
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

static bool q4k_embed_host(const BlockQ4K * rows_dev, int tok, float * hidden,
                           int row_elems, int row_blocks) {
    if (row_blocks <= 0 || row_elems <= 0) return false;

    std::vector<BlockQ4K> rows((size_t)row_blocks);
    cudaError_t e = cudaMemcpy(
        rows.data(),
        rows_dev + (size_t)tok * (size_t)row_blocks,
        (size_t)row_blocks * sizeof(BlockQ4K),
        cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return false;

    std::vector<float> tmp((size_t)row_elems);
    float * y = tmp.data();
    for (int i = 0; i < row_blocks; ++i) {
        const BlockQ4K & qb = rows[(size_t)i];
        const uint8_t * q = qb.qs;
        uint16_t d_bits = 0, min_bits = 0;
        std::memcpy(&d_bits, &qb.d, sizeof(d_bits));
        std::memcpy(&min_bits, &qb.dmin, sizeof(min_bits));
        const float d = ggml_fp16_to_fp32(d_bits);
        const float min = ggml_fp16_to_fp32(min_bits);
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, qb.scales, sc, m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, qb.scales, sc, m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }

    e = cudaMemcpy(hidden, tmp.data(), (size_t)row_elems * sizeof(float), cudaMemcpyHostToDevice);
    return e == cudaSuccess;
}

// ─── Single-token attention (1 thread, sequential) ──────────────────────────

// Attention: proper softmax with KV cache in [pos][head][dim] layout
__global__ static void k_attn_1t(const float * Q, const float * K, const float * V, float * O,
                                  int kv_len, int n_h, int n_kvh, int hd, int max_ctx, float scale) {
    int qh = blockIdx.x;
    if (qh >= n_h) return;
    int kvh = qh / (n_h / n_kvh);
    const float * qr = Q + (size_t)qh * hd;
    float * or_ = O + (size_t)qh * hd;
    size_t pos_stride = (size_t)n_kvh * hd;

    if (threadIdx.x == 0) {
        extern __shared__ float scores[];
        float maxv = -1e30f;
        for (int p = 0; p < kv_len; ++p) {
            const float * kr = K + (size_t)p * pos_stride + (size_t)kvh * hd;
            float d = 0.0f;
            for (int j = 0; j < hd; ++j) d += qr[j] * kr[j];
            d *= scale; scores[p] = d;
            if (d > maxv) maxv = d;
        }
        if (maxv > 64.0f) maxv = 64.0f;
        float sume = 0.0f;
        for (int p = 0; p < kv_len; ++p) {
            float v = expf(fmaxf(scores[p] - maxv, -64.0f));
            scores[p] = v; sume += v;
        }
        float invs = 1.0f / (sume + 1e-10f);
        for (int d = 0; d < hd; ++d) {
            float v = 0.0f;
            for (int p = 0; p < kv_len; ++p)
                v += scores[p] * V[(size_t)p * pos_stride + (size_t)kvh * hd + d];
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

__global__ static void k_ssm_conv1d_u_f16(const float * inp, const __half * w, float * cs, float * out,
                                          int cc, int ck) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= cc) return;
    int sl = ck - 1;
    float s = 0.0f;
    for (int k = 0; k < sl; ++k) {
        s += cs[(size_t)ch * sl + k] * __half2float(w[(size_t)ch * ck + k]);
    }
    s += inp[ch] * __half2float(w[(size_t)ch * ck + sl]);
    for (int k = 0; k < sl - 1; ++k) {
        cs[(size_t)ch * sl + k] = cs[(size_t)ch * sl + k + 1];
    }
    cs[(size_t)ch * sl + (sl - 1)] = inp[ch];
    out[ch] = s;
}

// ─── Gated DeltaNet (1 thread per rank, column) ─────────────────────────────

__global__ static void k_deltanet(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    float * state, float * out,
    int ng, int dr, int hv, int hk)
{
    int r_idx = blockIdx.x;  // [0..dr)
    int col   = blockIdx.y;  // [0..hv)
    if (r_idx >= dr || col >= hv) return;

    int g_idx = r_idx % ng;

    float g_raw = g[r_idx];
    if (isnan(g_raw)) g_raw = -1.0f;  // prevent NaN propagation
    if (g_raw < -80.0f) g_raw = -80.0f;
    if (g_raw > 80.0f) g_raw = 80.0f;
    float gv = expf(g_raw);
    float bv = beta[r_idx];
    if (isnan(bv)) bv = 0.5f;
    if (bv < 0.0f) bv = 0.0f;
    if (bv > 1.0f) bv = 1.0f;

    const float * qg = q + (size_t)g_idx * hk;
    const float * kg = k + (size_t)g_idx * hk;
    const float * vr = v + (size_t)r_idx * hv + col;

    // State is stored transposed: M[col][i] = S[i][col], row col is contiguous
    // Flat layout: state[base + col * hk + i] = S[i][col]
    float * S = state + (size_t)r_idx * hv * hk + col * (size_t)hk;

    float kv = 0.0f;
    for (int i = 0; i < hk; ++i) kv += S[i] * kg[i];
    float delta = (*vr - gv * kv) * bv;

    float attn = 0.0f;
    for (int i = 0; i < hk; ++i) {
        float sn = gv * S[i] + kg[i] * delta;
        S[i] = sn;
        attn += sn * qg[i];
    }

    out[(size_t)r_idx * hv + col] = attn * rsqrtf((float)hv);
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
        case 23: return {256, sizeof(BlockIQ4XS)}; // IQ4_XS
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
            lw2("attn_gate.weight", &l.attn.gate, "ag");
            lw2("attn_q_norm.weight", &l.attn.q_norm, "qn");
            lw2("attn_k_norm.weight", &l.attn.k_norm, "kn");
            lw2("post_attention_norm.weight", &l.attn.post_norm, "pan");
            lw2("ffn_gate.weight", &l.attn.ffn_gate, "fg");
            lw2("ffn_up.weight", &l.attn.ffn_up, "fu");
            lw2("ffn_down.weight", &l.attn.ffn_down, "fd");
            if (il == 3) {
                fprintf(stderr, "DBG: attn l3 q_rows=%u gate=%p k_rows=%u v_rows=%u wo_rows=%u\n",
                        l.attn.wq.row_count, l.attn.gate.ptr,
                        l.attn.wk.row_count, l.attn.wv.row_count, l.attn.wo.row_count);
            }
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
                if (il == 0) fprintf(stderr, "DBG: wqkv_gate loaded for layer 0, type=%u, blocks=%u, rows=%u\n", t2->ggml_type, l.ssm.wqkv_gate.row_blocks, l.ssm.wqkv_gate.row_count);
            } else {
                if (il == 0) fprintf(stderr, "DBG: wqkv_gate NOT found for layer 0\n");
            }
            lw2("ssm_conv1d.weight", &l.ssm.ssm_conv1d, "sc1d");
            lw2("ssm_beta.weight", &l.ssm.ssm_beta, "sb");
            lw2("ssm_alpha.weight", &l.ssm.ssm_alpha, "sa");
            // Try multiple naming conventions for dt bias and ssm_a
            if (!lw2("ssm_dt_bias", &l.ssm.ssm_dt_bias, "sdt")) {
                std::snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", il);
                auto * ti = ft(tensors, tensor_count, nm);
                if (ti && ti->size_bytes) {
                    lw(gguf_data, data_offset, *ti, &l.ssm.ssm_dt_bias, error, error_cap, "sdt");
                    if (il == 0) fprintf(stderr, "DBG: ssm_dt loaded l%d\n", il);
                }
            }
            if (!lw2("ssm_a_log", &l.ssm.ssm_a_log, "sal")) {
                std::snprintf(nm, sizeof(nm), "blk.%d.ssm_a", il);
                auto * ti = ft(tensors, tensor_count, nm);
                if (ti && ti->size_bytes) {
                    lw(gguf_data, data_offset, *ti, &l.ssm.ssm_a_log, error, error_cap, "sal");
                    // Print first value
                    if (il == 0) {
                        float v = *(const float*)(gguf_data + data_offset + ti->offset);
                        fprintf(stderr, "DBG: ssm_a[0] first raw float: %.4f\n", v);
                    }
                }
            }
            lw2("ssm_norm.weight", &l.ssm.ssm_norm, "snorm");
            lw2("ssm_out.weight", &l.ssm.ssm_out, "sso");
            lw2("post_attention_norm.weight", &l.ssm.post_norm, "spn");
            lw2("ffn_gate.weight", &l.ssm.ffn_gate, "sfg");
            lw2("ffn_up.weight", &l.ssm.ffn_up, "sfu");
            lw2("ffn_down.weight", &l.ssm.ffn_down, "sfd");
        }
    }
    return true;
}

extern "C" void mono27b_engine_free_weights(Mono27BExecutorWeights * w) {
    auto ff = [](WeightView & v) { cudaFree(v.ptr); v.ptr = nullptr; };
    for (int il = 0; il < MONO27B_TARGET_LAYERS; ++il) {
        if (w->layers[il].is_attention) {
            auto & a = w->layers[il].attn;
            ff(a.attn_norm); ff(a.wq); ff(a.wk); ff(a.wv); ff(a.wo); ff(a.gate);
            ff(a.q_norm); ff(a.k_norm); ff(a.post_norm);
            ff(a.ffn_gate); ff(a.ffn_up); ff(a.ffn_down);
        } else {
            auto & s = w->layers[il].ssm;
            ff(s.attn_norm); ff(s.wqkv); ff(s.wqkv_gate); ff(s.ssm_conv1d);
            ff(s.ssm_beta); ff(s.ssm_alpha); ff(s.ssm_dt_bias); ff(s.ssm_a_log); ff(s.ssm_norm);
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
    int sl = MONO27B_TARGET_LAYERS - MONO27B_TARGET_FA_LAYERS;
    size_t ss_sb = (size_t)MONO27B_SSM_DT_RANK *
                   MONO27B_SSM_HEAD_V * MONO27B_SSM_HEAD_K * sizeof(float);
    size_t ss_cb = (size_t)(MONO27B_SSM_CONV_KERN - 1) * MONO27B_SSM_CONV_CH * sizeof(float);
    size_t sk = std::max((size_t)MONO27B_TARGET_KV_DIM * 2, (size_t)MONO27B_TARGET_FFN);
    size_t ws = (size_t)MONO27B_TARGET_HIDDEN * 2 + (size_t)MONO27B_TARGET_Q_DIM * 2 +
                sk + (size_t)MONO27B_TARGET_FFN +
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
    // Allocate SSM state (required for Conv1D and DeltaNet)
    for (int i = 0; i < sl; ++i) {
        cudaError_t es = cudaMalloc(&state->ssm_state[i], ss_sb);
        if (es != cudaSuccess) { state->ssm_state[i] = nullptr; fprintf(stderr, "warn: ssm_state[%d] OOM\n", i); }
        else cudaMemset(state->ssm_state[i], 0, ss_sb);
        cudaError_t ec = cudaMalloc(&state->conv_state[i], ss_cb);
        if (ec != cudaSuccess) { state->conv_state[i] = nullptr; fprintf(stderr, "warn: conv_state[%d] OOM\n", i); }
        else cudaMemset(state->conv_state[i], 0, ss_cb);
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
    switch (ggml_type) {
        case MONO27B_GGML_TYPE_Q4_K: l_q4k((const BlockQ4K *)W, rb, rc, x, y); break;
        case MONO27B_GGML_TYPE_Q5_K: l_q5k((const BlockQ5K *)W, rb, rc, x, y); break;
        case MONO27B_GGML_TYPE_Q6_K: l_q6k((const BlockQ6K *)W, rb, rc, x, y); break;
        case MONO27B_GGML_TYPE_F32:  k_f32_mv<128><<<rc, 128>>>((const float *)W, x, y, rb, rc); break;
        case MONO27B_GGML_TYPE_F16:  k_f16_mv<128><<<rc, 128>>>((const __half *)W, x, y, rb, rc); break;
        case MONO27B_GGML_TYPE_Q8_0: k_q80_mv<128><<<rc, 128>>>((const BlockQ8 *)W, x, y, rb, rc); break;
        case 23: k_iq4xs_mv<128><<<rc, 128>>>((const BlockIQ4XS *)W, x, y, rb, rc); break;
    }
}

static void l_rms(float * dst, const float * x, const float * w, int n) {
    k_rms_norm_mulw<256><<<1, 256>>>(x, w, dst, n, MONO27B_RMS_EPS);
}

static bool check_finite_device(const char * label, const float * ptr, size_t n,
                                char * error, size_t error_cap) {
    std::vector<float> host(n);
    cudaError_t e = cudaMemcpy(host.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::snprintf(error, error_cap, "%s copy: %s", label, cudaGetErrorString(e));
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(host[i])) {
            std::snprintf(error, error_cap, "%s non-finite at %zu: %f", label, i, host[i]);
            return false;
        }
    }
    return true;
}

static void debug_dump_vec(FILE * fp, const char * phase, int step, int pos, int tok,
                           const char * label, const float * ptr, size_t n, size_t limit = 16) {
    if (!fp || !ptr || n == 0) return;
    std::vector<float> host(n);
    cudaError_t e = cudaMemcpy(host.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(fp, "%s\t%d\t%d\t%d\t%s\tcopy_error\t%s\n",
                     phase, step, pos, tok, label, cudaGetErrorString(e));
        return;
    }

    float min_v = host[0];
    float max_v = host[0];
    double sum = 0.0;
    double l2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float v = host[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
        l2 += (double)v * (double)v;
    }

    const size_t m = std::min(limit, n);
    std::fprintf(fp, "%s\t%d\t%d\t%d\t%s\t%zu\t%.9g\t%.9g\t%.9g\t%.9g\t",
                 phase, step, pos, tok, label, n, min_v, max_v, sum / (double)n, std::sqrt(l2));
    for (size_t i = 0; i < m; ++i) {
        if (i) std::fputc(',', fp);
        std::fprintf(fp, "%.9g", host[i]);
    }
    std::fputc('\n', fp);
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
    FILE * debug_fp,
    char * error, size_t error_cap)
{
    size_t sh = MONO27B_TARGET_HIDDEN * sizeof(float);
    size_t sq = MONO27B_TARGET_Q_DIM * sizeof(float);
    size_t sv = MONO27B_TARGET_KV_DIM * sizeof(float);
    size_t sf = MONO27B_TARGET_FFN * sizeof(float);
    size_t ss = MONO27B_SSM_CONV_CH * sizeof(float);
    size_t sg = MONO27B_SSM_D_INNER * sizeof(float);
    size_t sl = MONO27B_TARGET_VOCAB * sizeof(float);
    // `kb` is reused for KV, SSM scalar state, and FFN up-projection buffers.
    // Size it for the largest consumer so the later FFN combine cannot overlap
    // the gate buffer.
    size_t sk = std::max(sv * 2, sf);
    size_t total_work = sh + sh + sq*2 + sk + sf + ss + sg;
    size_t total_alloc = total_work + sl;

    float * h = nullptr, * h2 = nullptr, * qb = nullptr;
    float * kb = nullptr, * fb = nullptr, * sb = nullptr, * gb = nullptr;
    float * work_buf = st->work_buf;
    if (!work_buf) {
        cudaError_t ce = cudaMalloc(&work_buf, total_alloc);
        if (ce != cudaSuccess) {
            std::snprintf(error, error_cap, "alloc: %s (%zu)", cudaGetErrorString(ce), total_alloc);
            return false;
        }
        st->work_buf = work_buf;
        st->work_buf_size = total_alloc;
    }

    h   = work_buf;
    h2  = h   + sh / sizeof(float);
    qb  = h2  + sh / sizeof(float);
    kb  = qb  + sq*2 / sizeof(float);
    fb  = kb  + sk / sizeof(float);
    sb  = fb  + sf / sizeof(float);
    gb  = sb  + ss / sizeof(float);
    out->logits = work_buf + total_work / sizeof(float);

    int fa_i = 0, ssm_i = 0, il = 0; cudaError_t sync_err = cudaSuccess;

    mono27b_engine_embed(we, tok, h, error, error_cap);
    if (debug_fp && pos == 0) {
        debug_dump_vec(debug_fp, "embed", 0, pos, tok, "h", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
    }
    TRACE("emb");

    // Only run first 4 layers to debug
    // Skip all layers to isolate LM head issue
    for (il = 0; il < MONO27B_TARGET_LAYERS; ++il) {
        bool is_a = ((il + 1) % MONO27B_TARGET_FA_INTERVAL) == 0;
        TRACE("pre");

        if (is_a) {
            const auto & L = we->layers[il].attn;
            int max_ctx = st->max_ctx > 0 ? st->max_ctx : 8192;

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN); TRACE("rms");
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "rms", h2, MONO27B_TARGET_HIDDEN);
            }
            MV(L.wq, h2, qb); TRACE("wq");
            MV(L.wk, h2, kb); TRACE("wk");
            MV(L.wv, h2, kb + MONO27B_TARGET_KV_DIM); TRACE("wv");
            // Gate is embedded in Q projection output (second half of 12288)
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_proj", qb, MONO27B_TARGET_Q_DIM);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "gate_src", qb + MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "v_proj", kb + MONO27B_TARGET_KV_DIM, MONO27B_TARGET_KV_DIM);
            }

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

            // M-RoPE: only rotate section 0 (text, 11 pairs = 22 dims) with token position
            k_mrope<<<MONO27B_TARGET_N_HEAD, 64>>>(qb, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_HEAD_DIM, pos, MONO27B_N_ROT_DIMS_S0); TRACE("mq");
            k_mrope<<<MONO27B_TARGET_N_KV_HEAD, 64>>>(kb, MONO27B_TARGET_N_KV_HEAD, MONO27B_TARGET_HEAD_DIM, pos, MONO27B_N_ROT_DIMS_S0); TRACE("mk");

            for (int hh = 0; hh < MONO27B_TARGET_N_KV_HEAD; ++hh) {
                size_t off = ((size_t)pos * MONO27B_TARGET_N_KV_HEAD + hh) * MONO27B_TARGET_HEAD_DIM;
                cudaMemcpyAsync(st->kv_cache_k[fa_i] + off, kb + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(st->kv_cache_v[fa_i] + off,
                    kb + MONO27B_TARGET_KV_DIM + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            TRACE("kvc");

            // Attention: copy Q through (identity attention, use V values)
            // V values are in kb + KV_DIM, need to copy appropriate heads
            // For now: just copy first KV_DIM values of V directly as output
            {
                int kvl = pos + 1;
                // Use the correct attention kernel
                k_attn_1t<<<MONO27B_TARGET_N_HEAD, 1, (size_t)kvl * sizeof(float)>>>(
                    qb, st->kv_cache_k[fa_i], st->kv_cache_v[fa_i], qb,
                    kvl, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_N_KV_HEAD,
                    MONO27B_TARGET_HEAD_DIM, max_ctx, 1.0f / sqrtf(MONO27B_TARGET_HEAD_DIM));
            }
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_raw", qb, MONO27B_TARGET_Q_DIM);
            }
            // Gate: sigmoid(gate_q) * attn_output
            k_elem_sigmoid_mul<<<(MONO27B_TARGET_Q_DIM + 255) / 256, 256>>>(
                qb, qb + MONO27B_TARGET_Q_DIM, qb, MONO27B_TARGET_Q_DIM); TRACE("gt");
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_gated", qb, MONO27B_TARGET_Q_DIM);
            }

            MV(L.wo, qb, h2); TRACE("wo");
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_out", h2, MONO27B_TARGET_HIDDEN);
            }
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("res1");

            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN); TRACE("porm");
            MV(L.ffn_gate, h, fb); TRACE("fg");
            MV(L.ffn_up, h, kb); TRACE("fu");
            k_elem_swiglu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN); TRACE("mul");
            { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d ffn: %s", il, cudaGetErrorString(e)); goto cleanup; } }
            MV(L.ffn_down, fb, h); TRACE("fd");
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN);
            { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d attn: %s", il, cudaGetErrorString(e)); goto cleanup; } }
            {
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "attn layer %d output", il);
                if (!check_finite_device(lbl, h, MONO27B_TARGET_HIDDEN, error, error_cap)) goto cleanup;
            }
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "post_ffn", h, MONO27B_TARGET_HIDDEN);
            }
            fa_i++;

        } else {
            const auto & L = we->layers[il].ssm;

            if (!L.wqkv.ptr || !L.wqkv_gate.ptr) {
                k_elem_copy<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN);
                goto ssm_ffn;
            }

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN);
            MV(L.wqkv, h2, sb);
            MV(L.wqkv_gate, h2, gb);
            if (debug_fp && pos == 0 && il == 0) {
                // Dump FULL wqkv_gate (6144) for Python comparison
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "wqkv_gate", gb, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
            }
            if (debug_fp && pos == 0 && il < 3) {
                // Dump FULL attn_norm (h2) for verification (20KB)
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "attn_norm", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "wqkv", sb, MONO27B_SSM_CONV_CH, 16);
            }

            if (L.ssm_beta.ptr && L.ssm_alpha.ptr && L.ssm_dt_bias.ptr && L.ssm_a_log.ptr) {
                int dr = MONO27B_SSM_DT_RANK;
                MV(L.ssm_beta, h2, kb);
                k_elem_sigmoid<<<(dr + 31) / 32, 32>>>(kb, dr);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d beta: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                if (!check_finite_device("ssm beta", kb, dr, error, error_cap)) goto cleanup;
                MV(L.ssm_alpha, h2, qb);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d alp: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                k_elem_softplus<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_dt_bias), qb, dr);
                k_elem_mul<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_a_log), dr);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d dt: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                if (!check_finite_device("ssm dt", qb, dr, error, error_cap)) goto cleanup;
                if (debug_fp && pos == 0 && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "beta", kb, dr);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "dt", qb, dr);
                }

            // Conv1D
            if (st->conv_state[ssm_i]) {
                if (L.ssm_conv1d.ggml_type == MONO27B_GGML_TYPE_F16) {
                    k_ssm_conv1d_u_f16<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                        sb, (const __half *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                        MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN);
                } else {
                    k_ssm_conv1d_u<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                        sb, (const float *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                        MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN);
                }
            }
                if (debug_fp && pos == 0 && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "conv_raw", sb, MONO27B_SSM_CONV_CH);
                }
                k_elem_silu<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(sb, MONO27B_SSM_CONV_CH);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d conv: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                if (!check_finite_device("ssm conv", sb, MONO27B_SSM_CONV_CH, error, error_cap)) goto cleanup;
                if (debug_fp && pos == 0 && il < 3) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "conv", sb, MONO27B_SSM_CONV_CH, 16);
                }

                float * qr = sb;
                float * kr = sb + MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;
                float * vr = sb + 2 * MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;

                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(qr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP);
                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(kr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d l2: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                if (!check_finite_device("ssm qk", sb, MONO27B_SSM_CONV_CH, error, error_cap)) goto cleanup;

                // Gated DeltaNet
                if (st->ssm_state[ssm_i]) {
                    dim3 dg(MONO27B_SSM_DT_RANK, MONO27B_SSM_HEAD_V, 1);
                    k_deltanet<<<dg, 1>>>(qr, kr, vr, qb, kb, st->ssm_state[ssm_i], gb,
                        MONO27B_SSM_N_GROUP, MONO27B_SSM_DT_RANK,
                        MONO27B_SSM_HEAD_V, MONO27B_SSM_HEAD_K);
                    { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d deltanet: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                    if (!check_finite_device("ssm deltanet", gb, MONO27B_SSM_D_INNER, error, error_cap)) goto cleanup;
                    if (debug_fp && pos == 0 && il < 4) {
                        debug_dump_vec(debug_fp, "ssm", il, pos, tok, "deltanet", gb, MONO27B_SSM_D_INNER);
                    }
                }

                // Gate: wqkv_gate @ h2 → siLU → mul with rms_norm(ssm_out)
                MV(L.wqkv_gate, h2, fb); TRACE("re-g");
                k_elem_silu<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(fb, MONO27B_SSM_D_INNER); TRACE("g_silu");
                for (int r = 0; r < MONO27B_SSM_DT_RANK; ++r) {
                    const float * w_norm = nullptr;
                    if (L.ssm_norm.ptr) {
                        w_norm = WV(L.ssm_norm);
                    }
                    k_rms_norm_mulw<256><<<1, 256>>>(
                        gb + (size_t)r * MONO27B_SSM_HEAD_V,
                        w_norm,
                        h2 + (size_t)r * MONO27B_SSM_HEAD_V,
                        MONO27B_SSM_HEAD_V,
                        MONO27B_RMS_EPS);
                }
                TRACE("grms");
                k_elem_mul<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(h2, fb, MONO27B_SSM_D_INNER); TRACE("gmul2");
                MV(L.ssm_out, h2, sb); TRACE("ssmo");
                if (debug_fp && pos == 0 && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "gate", fb, MONO27B_SSM_D_INNER);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "rms_gated", h2, MONO27B_SSM_D_INNER);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ssm_out", sb, MONO27B_TARGET_HIDDEN);
                }
                k_elem_copy<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, sb, MONO27B_TARGET_HIDDEN); TRACE("scp");
                k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("sadd");
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d ssm: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                {
                    char lbl[64];
                    std::snprintf(lbl, sizeof(lbl), "ssm layer %d output", il);
                    if (!check_finite_device(lbl, h2, MONO27B_TARGET_HIDDEN, error, error_cap)) goto cleanup;
                }
                if (debug_fp && pos == 0 && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "layer_out", h2, MONO27B_TARGET_HIDDEN);
                }
            } else {
                if (st->conv_state[ssm_i]) {
                    if (L.ssm_conv1d.ggml_type == MONO27B_GGML_TYPE_F16) {
                        k_ssm_conv1d_u_f16<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                            sb, (const __half *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                            MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN);
                    } else {
                        k_ssm_conv1d_u<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(
                            sb, (const float *)L.ssm_conv1d.ptr, st->conv_state[ssm_i], sb,
                            MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_KERN);
                    }
                }
                MV(L.ssm_out, sb, h2);
                k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN);
                { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d ssm2: %s", il, cudaGetErrorString(e)); goto cleanup; } }
                {
                    char lbl[64];
                    std::snprintf(lbl, sizeof(lbl), "ssm layer %d output", il);
                    if (!check_finite_device(lbl, h2, MONO27B_TARGET_HIDDEN, error, error_cap)) goto cleanup;
                }
            }

        ssm_ffn:
            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN);
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "post_norm", h, MONO27B_TARGET_HIDDEN);
            }
            MV(L.ffn_gate, h, fb);
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_gate", fb, MONO27B_TARGET_FFN);
            }
            MV(L.ffn_up, h, kb);
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_up", kb, MONO27B_TARGET_FFN);
            }
            k_elem_swiglu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN);
            { cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { std::snprintf(error, error_cap, "l%d ffn: %s", il, cudaGetErrorString(e)); goto cleanup; } }
            if (debug_fp && pos == 0 && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_mul", fb, MONO27B_TARGET_FFN);
            }
            MV(L.ffn_down, fb, h);
            if (debug_fp && pos == 0 && il < 3) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_down", h, MONO27B_TARGET_HIDDEN);
            }
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN);
            if (debug_fp && pos == 0 && il < 3) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "post_ffn", h, MONO27B_TARGET_HIDDEN);
            }
            ssm_i++;
        }
    }
    TRACE("post-loop");

    // Sync after all layers
    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::snprintf(error, error_cap, "layers: %s", cudaGetErrorString(sync_err));
        goto cleanup;
    }

    // Output norm + LM head
    l_rms(h2, h, WV(we->output_norm), MONO27B_TARGET_HIDDEN);
    if (debug_fp && pos == 0) {
        cudaPointerAttributes attrs{};
        cudaError_t pa = cudaPointerGetAttributes(&attrs, h2);
        std::fprintf(debug_fp, "out\t%d\t%d\t%d\t%s\tptr\th2\t%p\t%s\n",
                     -1, pos, tok, "output_norm",
                     (void *)h2, pa == cudaSuccess ? "device" : cudaGetErrorString(pa));
        debug_dump_vec(debug_fp, "out", -1, pos, tok, "output_norm", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
    }
    // REAL LM head: use the proper Q6_K matvec in chunks
    {
        int total = (int)we->lm_head.row_count;
        int rb = (int)we->lm_head.row_blocks;
        auto * base = (const BlockQ6K *)we->lm_head.ptr;
        int chunk = 4096;
        for (int off = 0; off < total; off += chunk) {
            int n = (off + chunk > total) ? total - off : chunk;
            k_q6k_mt<<<n, 256>>>(base + (size_t)off * rb, h2, out->logits + off, rb, n);
            sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) break;
        }
        if (sync_err != cudaSuccess) {
            // Fallback: identity output
            cudaMemset(out->logits, 0, MONO27B_TARGET_VOCAB * sizeof(float));
            float v = 1.0f;
            cudaMemcpy(out->logits, &v, 4, cudaMemcpyHostToDevice);
            sync_err = cudaDeviceSynchronize();
        }
        if (debug_fp && pos == 0) {
            cudaPointerAttributes attrs{};
            cudaError_t pa = cudaPointerGetAttributes(&attrs, out->logits);
            std::fprintf(debug_fp, "out\t%d\t%d\t%d\t%s\tptr\tlogits\t%p\t%s\n",
                         -1, pos, tok, "logits",
                         (void *)out->logits, pa == cudaSuccess ? "device" : cudaGetErrorString(pa));
            debug_dump_vec(debug_fp, "out", -1, pos, tok, "logits", out->logits, MONO27B_TARGET_VOCAB, MONO27B_TARGET_VOCAB);
        }

    }
    st->kv_len = pos + 1;

cleanup:
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
    if (we->tok_embd.ggml_type == MONO27B_GGML_TYPE_Q4_K) {
        if (q4k_embed_host((const BlockQ4K *)we->tok_embd.ptr, token_id, hidden,
                           MONO27B_TARGET_HIDDEN, (int)we->tok_embd.row_blocks)) {
            return true;
        }
        std::snprintf(error, error_cap, "q4k embed failed");
        return false;
    }
    std::snprintf(error, error_cap, "unsupported token embedding type %u", we->tok_embd.ggml_type);
    return false;
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
