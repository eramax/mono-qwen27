#include "mono27b_config.h"
#include "mono27b_gguf.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

bool g_mono27b_quiet = false;

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

// Q8_1 block for intermediate matvec quantization (32 elements per block)
struct BlockQ8_1 {
    union {
        struct { __half d; __half s; } ds_alias;
        __half2 ds;
    };
    int8_t qs[32];
};

// Quantize F32 vector to Q8_1 blocks in a global buffer
// Matches ggml's quantize_q8_1 kernel exactly (roundf, no clamp, sequential per-thread)
__global__ static void k_quant_q8_1(const float * x, BlockQ8_1 * y, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int nb = n / 32;
    if (b >= nb) return;
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
        float v = x[(size_t)b * 32 + j];
        amax = fmaxf(amax, fabsf(v));
    }
    float d = amax / 127.0f;
    float sum_ = 0.0f;
    for (int j = 0; j < 32; j++) {
        int qi = (amax == 0.0f) ? 0 : (int)(roundf(x[(size_t)b * 32 + j] / d));
        y[b].qs[j] = (int8_t)qi;
        sum_ += (float)qi;
    }
    y[b].ds = make_half2(d, d * sum_);
}

// ─── dp4a helper (from ggml common.cuh) ─────────────────────────────────────

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= 610 || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return x16[2*i32 + 0] | (x16[2*i32 + 1] << 16);
}

static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const float * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

// ─── Q4_K vec_dot (from vecdotq.cuh, vmmq path) ─────────────────────────────

#define MONO27B_QK8_1 32
#define MONO27B_QI8_1 (MONO27B_QK8_1 / 4)
#define MONO27B_QR4_K 2
#define MONO27B_QR5_K 2
#define MONO27B_QR6_K 2
#define MONO27B_QI6_K 32

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const __half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < MONO27B_QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const BlockQ8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const BlockQ4K * bq4_K = (const BlockQ4K *) vbq + kbx;

    int    v[2];
    int    u[2*MONO27B_QR4_K];
    float d8[MONO27B_QR4_K];

    const int bq8_offset = MONO27B_QR4_K * ((iqs/2) / (MONO27B_QI8_1/2));
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < MONO27B_QR4_K; ++i) {
        const BlockQ8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, *(const __half2 *)&bq4_K->d, d8);
}

__global__ static void k_q4k_mv_q8_dp4a(const BlockQ4K * W, const BlockQ8_1 * q8, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[128];
    float sum = 0.0f;
    const BlockQ4K * wp = W + (size_t)row * rb;
    for (int idx = threadIdx.x; idx < rb * 16; idx += 128) {
        int b = idx / 16;
        int iqs = (idx % 16) * 2;
        sum += vec_dot_q4_K_q8_1(wp, q8 + b * 8, b, iqs);
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── Q5_K vec_dot (from vecdotq.cuh, vmmq path) ─────────────────────────────

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const __half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < MONO27B_QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = __half22float2(dm5);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const BlockQ8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const BlockQ5K * bq5_K = (const BlockQ5K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*MONO27B_QR5_K];
    float d8[MONO27B_QR5_K];

    const int bq8_offset = MONO27B_QR5_K * ((iqs/2) / (MONO27B_QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < MONO27B_QR5_K; ++i) {
        const BlockQ8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, *(const __half2 *)&bq5_K->d, d8);
}

__global__ static void k_q5k_mv_q8(const BlockQ5K * W, const BlockQ8_1 * q8, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[128];
    float sum = 0.0f;
    const BlockQ5K * wp = W + (size_t)row * rb;
    for (int idx = threadIdx.x; idx < rb * 16; idx += 128) {
        int b = idx / 16;
        int iqs = (idx % 16) * 2;
        sum += vec_dot_q5_K_q8_1(wp, q8 + b * 8, b, iqs);
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── Q6_K vec_dot (from vecdotq.cuh, mmvq path) ─────────────────────────────

#define MONO27B_QR6_K 2

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < MONO27B_QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const BlockQ8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const BlockQ6K * bq6_K = (const BlockQ6K *) vbq + kbx;

    const int bq8_offset = 2 * MONO27B_QR6_K * (iqs / (MONO27B_QI6_K/2)) + (iqs % (MONO27B_QI6_K/2)) / (MONO27B_QI6_K/4);
    const int scale_offset = (MONO27B_QI6_K/4) * (iqs / (MONO27B_QI6_K/2)) + (iqs % (MONO27B_QI6_K/2)) / (MONO27B_QI6_K/8);
    const int vh_shift = 2 * ((iqs % (MONO27B_QI6_K/2)) / (MONO27B_QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (MONO27B_QI6_K/4) * (iqs / (MONO27B_QI6_K/2)) + iqs % (MONO27B_QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[MONO27B_QR6_K];
    float d8[MONO27B_QR6_K];

#pragma unroll
    for (int i = 0; i < MONO27B_QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % MONO27B_QI8_1);
        d8[i] = __half2float(bq8_1[bq8_offset + 2*i].ds_alias.d);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

__global__ static void k_q6k_mv_q8_dp4a(const BlockQ6K * W, const BlockQ8_1 * q8, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[128];
    float sum = 0.0f;
    const BlockQ6K * wp = W + (size_t)row * rb;
    for (int idx = threadIdx.x; idx < rb * 32; idx += 128) {
        int b = idx / 32;
        int iqs = idx % 32;
        sum += vec_dot_q6_K_q8_1(wp, q8 + b * 8, b, iqs);
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ─── IQ4_XS vec_dot (from vecdotq.cuh, mmvq path) ───────────────────────────

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const BlockQ8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const BlockIQ4XS * bq4 = (const BlockIQ4XS *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

      const float d = __half2float(bq4->d) * __half2float(bq8_1[iqs/4].ds_alias.d);
    return d * sumi;
}

__global__ static void k_iq4xs_mv_q8_dp4a(const BlockIQ4XS * W, const BlockQ8_1 * q8, float * y, int rb, int rc) {
    int row = blockIdx.x;
    if (row >= rc) return;
    __shared__ float sh[128];
    float sum = 0.0f;
    const BlockIQ4XS * wp = W + (size_t)row * rb;
    for (int idx = threadIdx.x; idx < rb * 8; idx += 128) {
        int b = idx / 8;
        int iqs = (idx % 8) * 4;
        sum += vec_dot_iq4_xs_q8_1(wp, q8 + b * 8, b, iqs);
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ static void k_elem_softplus(const float * x, const float * bias, float * out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float z = x[i] + bias[i];
        out[i] = (z > 20.0f) ? z : logf(1.0f + expf(z));
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
        out[i] = (out[i] / (1.0f + expf(-out[i]))) * x;
    }
}

// Qwen3Next: wq outputs interleaved [Q0(256), G0(256), Q1(256), G1(256), ...]
// Deinterleave into contiguous Q [Q0..Q23] and contiguous gate [G0..G23].
// Grid = n_head blocks, block = head_dim threads.
__global__ static void k_deinterleave_qg(const float * src, float * q_dst, float * g_dst, int n_head, int head_dim) {
    int hi = blockIdx.x;
    int h  = threadIdx.x;
    if (hi >= n_head || h >= head_dim) return;
    q_dst[hi * head_dim + h] = src[hi * (head_dim * 2) + h];
    g_dst[hi * head_dim + h] = src[hi * (head_dim * 2) + head_dim + h];
}

// ─── RMS norm ────────────────────────────────────────────────────────────────



// Match ggml's reference RMS/L2 accumulation order more closely by using
// a single-thread sequential sum in double precision.
template <int BLK>
__global__ static void k_rms_norm_mulw(const float * x, const float * w, float * y, int n, float eps) {
    __shared__ double sh[BLK];
    double sum = 0.0;
    for (int i = threadIdx.x; i < n; i += BLK) {
        double val = (double)x[i];
        sum += val * val;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    double total_sum = sh[0];
    const double scale = 1.0 / sqrt(total_sum / (double)n + (double)eps);
    for (int i = threadIdx.x; i < n; i += BLK) {
        y[i] = (float)((double)x[i] * scale * (double)(w ? w[i] : 1.0f));
    }
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
        float d = __half2float(qb.d);
        float mn = __half2float(qb.dmin);
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
            sum += q6k_val(qb, e) * x[(size_t)b * 256 + e];
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
    float * x = d + (size_t)g * gs;
    __shared__ double sh[128];
    double s = 0.0;
    for (int i = threadIdx.x; i < gs; i += 128) s += (double)x[i] * (double)x[i];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int step = 64; step > 0; step >>= 1) {
        if (threadIdx.x < step) sh[threadIdx.x] += sh[threadIdx.x + step];
        __syncthreads();
    }
    double sum_sq = sh[0];
    // Match PyTorch/ggml reference: scale = 1 / sqrt(max(sum_sq, eps*eps))
    const float scale = rsqrtf(fmaxf((float)sum_sq, MONO27B_RMS_EPS * MONO27B_RMS_EPS));
    for (int i = threadIdx.x; i < gs; i += 128) x[i] *= scale;
}

// ─── M-RoPE ──────────────────────────────────────────────────────────────────
// Qwen3.5 uses multi-section RoPE with 4 position streams.
// For text tokens, llama.cpp expands a 1D token position into:
//   pos[0] = token_pos
//   pos[1] = token_pos
//   pos[2] = token_pos
//   pos[3] = 0
// and applies RoPE across all 64 rotary dims using sections [11, 11, 10, 0]
// in pair units.

// Device-side parameter pointers for CUDA graph replay.
// Initialize once, update device memory between replays.
__device__ int g_mrope_pos_t = 0;
__device__ int g_mrope_pos_h = 0;
__device__ int g_mrope_pos_w = 0;
__device__ int g_mrope_pos_x = 0;
__device__ int g_attn_kv_len = 0;
__device__ int g_embed_token_id = 0;

__global__ static void k_mrope(
    float * buf, int n_heads, int head_dim,
    int pos_t, int pos_h, int pos_w, int pos_x,
    int sec0, int sec1, int sec2, int sec3,
    int n_rot_dims)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float * hd = buf + (size_t)h * head_dim;

    const int sec01 = sec0 + sec1;
    const int sec012 = sec01 + sec2;
    const int n_rot_pairs = n_rot_dims / 2;
    const int half = n_rot_pairs;

    for (int p = threadIdx.x; p < n_rot_pairs; p += blockDim.x) {
        int d0 = p;
        int d1 = p + half;
        if (d1 >= head_dim) break;

        int pos = pos_x;
        if (p < sec0) pos = pos_t;
        else if (p < sec01) pos = pos_h;
        else if (p < sec012) pos = pos_w;

        float theta = powf(MONO27B_TARGET_ROPE_THETA, -2.0f * (float)p / (float)n_rot_dims);
        float c = cosf(theta * (float)pos), s = sinf(theta * (float)pos);
        float v0 = hd[d0], v1 = hd[d1];
        hd[d0] = v0 * c - v1 * s;
        hd[d1] = v0 * s + v1 * c;
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

// ─── Parallel attention: cooperative dot product, softmax, weighted sum ─────
// Each block handles one query head. All 256 threads cooperate on dot products,
// softmax reduction, and output accumulation. This replaces the old single-thread
// kernel which was both slow and numerically different from flash attention.
__global__ static void k_attn_parallel(
    const float * __restrict__ Q, const float * __restrict__ K,
    const float * __restrict__ V, float * __restrict__ O,
    int kv_len, int n_h, int n_kvh, int hd, int max_ctx, float scale) {

    int qh = blockIdx.x;
    if (qh >= n_h) return;
    int kvh = qh / (n_h / n_kvh);
    const float * qr = Q + (size_t)qh * hd;
    float * or_ = O + (size_t)qh * hd;
    size_t pos_stride = (size_t)n_kvh * hd;
    int t = threadIdx.x;
    // Shared memory: scores[kv_len] + max_val[1] + sum_exp[1] + scratch[256]
    extern __shared__ float smem[];
    float * scores  = smem;
    float * s_max   = smem + kv_len;
    float * s_sum   = s_max + 1;
    float * scratch = s_sum + 1;

    // Each thread loads one element of Q (reused across all positions)
    float q_elem = qr[t];

    // Phase 1: compute dot products and find max
    float maxv = -1e30f;
    for (int p = 0; p < kv_len; ++p) {
        const float * kr = K + (size_t)p * pos_stride + (size_t)kvh * hd;
        float d = q_elem * kr[t];
        scratch[t] = d;
        // Block reduction (tree) to sum all partial dots
        for (int s = hd / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (t < s) scratch[t] += scratch[t + s];
        }
        __syncthreads();
        if (t == 0) {
            float dot = scratch[0] * scale;
            scores[p] = dot;
            if (dot > maxv) maxv = dot;
        }
        __syncthreads();
    }
    __syncthreads();

    // Broadcast max to all threads via shared memory
    if (t == 0) {
        if (maxv > 64.0f) maxv = 64.0f;
        s_max[0] = maxv;
    }
    __syncthreads();
    maxv = s_max[0];

    // Phase 2: softmax — compute exp(s - max) and sum
    float local_sum = 0.0f;
    for (int p = t; p < kv_len; p += blockDim.x) {
        float v = expf(fmaxf(scores[p] - maxv, -64.0f));
        scores[p] = v;
        local_sum += v;
    }
    // Reduce partial sums
    scratch[t] = local_sum;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (t < s) scratch[t] += scratch[t + s];
    }
    __syncthreads();
    if (t == 0) s_sum[0] = scratch[0];
    __syncthreads();
    float inv_sum = 1.0f / (s_sum[0] + 1e-10f);

    // Phase 3: weighted sum of V values
    float out_val = 0.0f;
    for (int p = 0; p < kv_len; ++p) {
        float sv = scores[p] * inv_sum;
        out_val += sv * V[(size_t)p * pos_stride + (size_t)kvh * hd + t];
    }
    or_[t] = out_val;
}

// ─── SSM conv1d (1 thread per channel) ───────────────────────────────────────

__global__ static void k_ssm_conv1d_u(const float * inp, const float * w, float * cs, float * out,
                                       int cc, int ck) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= cc) return;
    int sl = ck - 1;
    float s = 0.0f;
    // Store history oldest-first to match ggml's streaming conv layout.
    // cs[0] is the oldest previous token, cs[sl-1] the most recent previous token.
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
    // Store history oldest-first to match ggml's streaming conv layout.
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

// ─── Gated DeltaNet (ggml-style warp-level parallelism) ──────────────────────
// Ported from llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu
// Uses warp-level primitives for exact numerical parity.

template <int S_v, bool KDA>
__global__ void __launch_bounds__(128, 2)
k_deltanet_ggml(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    const float * curr_state,
    float * out, float * state,
    int ng, int dr, int hv, int hk)
{
    const uint32_t h_idx = blockIdx.x;   // rank index [0, dr)
    const int      lane  = threadIdx.x;  // [0, 31)
    const int      col   = blockIdx.z * blockDim.y + threadIdx.y;  // [0, hv)

    if (h_idx >= dr || col >= hv) return;

    const int g_idx = h_idx % ng;  // group index for shared q/k

    const size_t state_offset = (size_t)h_idx * hv * hk;
    const float * cs = curr_state + state_offset + col * hk;
    float *       ws = state + state_offset;

    const float * qg = q + (size_t)g_idx * hk;
    const float * kg = k + (size_t)g_idx * hk;
    const float * vr = v + (size_t)h_idx * hv + col;

    const float beta_val = beta[h_idx];
    const float * g_ptr  = g + h_idx;

    constexpr int warp_size = 32;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;
    float s_shard[rows_per_lane];

    // Load state (transposed: S[i][col] is contiguous in col)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r] = cs[i];
    }

    if constexpr (!KDA) {
        const float g_val = expf(*g_ptr);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = kg[i];
            q_reg[r] = qg[i];
        }

        // kv[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = (*vr - g_val * kv_col) * beta_val;

        // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
        // attn[col] = sum_i S[i][col] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = warp_reduce_sum<warp_size>(attn_partial);

        if (lane == 0) {
            out[(size_t)h_idx * hv + col] = attn_col * (1.0f / sqrtf((float)hk));
        }
    } else {
        // KDA path (g is per-element vector, not used in Mono27B)
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = kg[i];
            q_reg[r] = qg[i];
        }

        // kv[col] = sum_i exp(g[i]) * S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            kv_shard += expf(g_ptr[i]) * s_shard[r] * k_reg[r];
        }

        float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        // delta[col] = (v[col] - kv[col]) * beta
        float delta_col = (*vr - kv_col) * beta_val;

        // fused: S[i][col] = exp(g[i]) * S[i][col] + k[i] * delta[col]
        // attn[col] = sum_i S[i][col] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r] = expf(g_ptr[i]) * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = warp_reduce_sum<warp_size>(attn_partial);

        if (lane == 0) {
            out[(size_t)h_idx * hv + col] = attn_col * (1.0f / sqrtf((float)hk));
        }
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        ws[col * hk + i] = s_shard[r];
    }
}

static void k_deltanet_ggml_launch(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    const float * curr_state,
    float * out, float * state,
    int ng, int dr, int hv, int hk)
{
    constexpr int warp_size = 32;
    constexpr int num_warps = 4;
    constexpr int S_v = 128;  // MONO27B_SSM_HEAD_V / HEAD_K

    dim3 grid_dims(dr, 1, (S_v + num_warps - 1) / num_warps);
    dim3 block_dims(warp_size, num_warps, 1);

    // Our model uses scalar g (KDA=false)
    k_deltanet_ggml<S_v, false><<<grid_dims, block_dims>>>(
        q, k, v, g, beta, curr_state, out, state,
        ng, dr, hv, hk);
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

    // Allocate Q8_1 scratch buffer for matvec quantization
    { size_t q8_sz = 544 * sizeof(BlockQ8_1);
      state->q8_scratch = nullptr;
      cudaError_t eq = cudaMalloc(&state->q8_scratch, q8_sz);
      if (eq != cudaSuccess) { state->q8_scratch = nullptr; } }

    // CUDA graph state: initially empty
    state->graph = nullptr;
    state->graphExec = nullptr;
    state->graph_captured = false;

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
    if (state->q8_scratch) cudaFree(state->q8_scratch);
    if (state->graphExec) { cudaGraphExecDestroy((cudaGraphExec_t)state->graphExec); state->graphExec = nullptr; }
    if (state->graph) { cudaGraphDestroy((cudaGraph_t)state->graph); state->graph = nullptr; }
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

// File-scope pointer to Q8_1 scratch buffer (set by engine before decode step)
static BlockQ8_1 * g_q8_scratch = nullptr;

// Forward declarations
static void l_mv_q8(void * W, uint32_t ggml_type, int rb, int rc, float * y);
static void l_mv_fallback(void * W, uint32_t ggml_type, int rb, int rc, const float * x, float * y);

// Quantize x to Q8_1 in the global scratch buffer; returns n_q8 blocks used
static int l_quant_q8(const float * x, int rb) {
    int n_q8 = rb * 8;
    if (g_q8_scratch && n_q8 <= 544) {
        k_quant_q8_1<<<(n_q8 + 127) / 128, 128>>>(x, g_q8_scratch, n_q8 * 32);
    }
    return n_q8;
}

// Single matvec: quantizes input, launches matvec (convenience wrapper)
static void l_mv_quant(void * W, uint32_t ggml_type, int rb, int rc, const float * x, float * y) {
    if (rc == 0 || !W) return;
    // Try Q8_1 dp4a path for K-quant types (fast).
    // For F32/F16/Q8_0/etc., fall through to the f32 dequant fallback.
    if (g_q8_scratch && rb * 8 <= 544) {
        switch (ggml_type) {
            case MONO27B_GGML_TYPE_Q4_K:
            case MONO27B_GGML_TYPE_Q5_K:
            case MONO27B_GGML_TYPE_Q6_K:
            case 23:
                l_quant_q8(x, rb);
                l_mv_q8(W, ggml_type, rb, rc, y);
                return;
            default: break;
        }
    }
    l_mv_fallback(W, ggml_type, rb, rc, x, y);
}

// Matvec using ALREADY QUANTIZED Q8_1 data in g_q8_scratch
static void l_mv_q8(void * W, uint32_t ggml_type, int rb, int rc, float * y) {
    if (rc == 0 || !W) return;
    if (!g_q8_scratch || rb * 8 > 544) return;
    switch (ggml_type) {
        case MONO27B_GGML_TYPE_Q4_K:
            k_q4k_mv_q8_dp4a<<<rc, 128>>>((const BlockQ4K *)W, g_q8_scratch, y, rb, rc);
            return;
        case MONO27B_GGML_TYPE_Q5_K:
            k_q5k_mv_q8<<<rc, 128>>>((const BlockQ5K *)W, g_q8_scratch, y, rb, rc);
            return;
        case MONO27B_GGML_TYPE_Q6_K:
            k_q6k_mv_q8_dp4a<<<rc, 128>>>((const BlockQ6K *)W, g_q8_scratch, y, rb, rc);
            return;
        case 23:
            k_iq4xs_mv_q8_dp4a<<<rc, 128>>>((const BlockIQ4XS *)W, g_q8_scratch, y, rb, rc);
            return;
        default: break;
    }
}

// Fallback path (F32 dequant)
static void l_mv_fallback(void * W, uint32_t ggml_type, int rb, int rc, const float * x, float * y) {
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

#define MV(wv, x, y) l_mv_quant((wv).ptr, (wv).ggml_type, (wv).row_blocks, (wv).row_count, x, y)
#define MV_Q8(wv, y) l_mv_q8((wv).ptr, (wv).ggml_type, (wv).row_blocks, (wv).row_count, y)
#define WV(wv) ((const float *)(wv).ptr)
#define TRACE(lbl) ((void)0)

// Production-performance CHECK_FINITE: skip finite checks for speed.
// In debug mode, include them to catch NaN/Inf early.
#ifdef MONO27B_DEBUG_FINITE
#define CHECK_FINITE(label, ptr, n) do { \
    if (!check_finite_device(label, ptr, n, error, error_cap)) goto cleanup; \
} while(0)
#define CHECK_FINITE_FMT(fmt, arg, ptr, n) do { \
    char _lbl[64]; std::snprintf(_lbl, sizeof(_lbl), fmt, arg); \
    if (!check_finite_device(_lbl, ptr, n, error, error_cap)) goto cleanup; \
} while(0)
#else
#define CHECK_FINITE(label, ptr, n) ((void)0)
#define CHECK_FINITE_FMT(fmt, arg, ptr, n) ((void)0)
#endif

extern "C" bool mono27b_engine_decode_step(
    const Mono27BExecutorWeights * we,
    Mono27BExecutorState * st,
    int tok, int pos,
    Mono27BLogitsOutput * out,
    FILE * debug_fp,
    int debug_pos,
    char * error, size_t error_cap)
{
    // Set Q8_1 scratch pointer for matvec quantization
    g_q8_scratch = (BlockQ8_1 *)st->q8_scratch;
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
    const bool dump_step = debug_fp && (debug_pos == -1 || pos == debug_pos);

    mono27b_engine_embed(we, tok, h, error, error_cap);
    if (dump_step) {
        debug_dump_vec(debug_fp, "embed", 0, pos, tok, "h", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
    }
    TRACE("emb");

    // CUDA graph replay is FUTURE WORK.
    // The KV cache write position changes each step, making graph capture
    // impractical without kernel modifications. Disabled for now.
    bool capture_started = false;

    // Only run first 4 layers to debug
    // Skip all layers to isolate LM head issue
    for (il = 0; il < MONO27B_TARGET_LAYERS; ++il) {
        bool is_a = ((il + 1) % MONO27B_TARGET_FA_INTERVAL) == 0;
        TRACE("pre");

        if (is_a) {
            const auto & L = we->layers[il].attn;
            int max_ctx = st->max_ctx > 0 ? st->max_ctx : 8192;

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN); TRACE("rms");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "rms", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            MV(L.wq, h2, qb); TRACE("wq");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_raw", qb, MONO27B_TARGET_Q_DIM * 2, MONO27B_TARGET_Q_DIM * 2);
            }
            // Qwen3Next: Q projection outputs interleaved Q+gate per head:
            // [Q0(256), G0(256), Q1(256), G1(256), ...]
            // Deinterleave to contiguous Q (6144) and gate (6144).
            // Use kb as a temporary source buffer to avoid in-place race
            // (head N overwrites src data that head 0..N-1 still need to read).
            {
                cudaMemcpyAsync(kb, qb, MONO27B_TARGET_Q_DIM * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
                int n_h = MONO27B_TARGET_N_HEAD;
                int hd = MONO27B_TARGET_HEAD_DIM;
                k_deinterleave_qg<<<n_h, hd>>>(kb, qb, qb + MONO27B_TARGET_Q_DIM, n_h, hd);
            }
            MV(L.wk, h2, kb); TRACE("wk");
            MV(L.wv, h2, kb + MONO27B_TARGET_KV_DIM); TRACE("wv");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_proj", qb, MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "gate_src", qb + MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM);
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
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_norm", qb, MONO27B_TARGET_Q_DIM, 64);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "k_norm", kb, MONO27B_TARGET_KV_DIM, 64);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_conv_predelta", qb, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "k_conv_predelta", kb, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K);
            }
            TRACE("qkn");

            // M-RoPE: text tokens use the same token position for the first
            // three position streams and 0 for the 4th stream.
            k_mrope<<<MONO27B_TARGET_N_HEAD, 64>>>(
                qb, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_HEAD_DIM,
                pos, pos, pos, 0,
                11, 11, 10, 0,
                MONO27B_N_ROT_DIMS); TRACE("mq");
            k_mrope<<<MONO27B_TARGET_N_KV_HEAD, 64>>>(
                kb, MONO27B_TARGET_N_KV_HEAD, MONO27B_TARGET_HEAD_DIM,
                pos, pos, pos, 0,
                11, 11, 10, 0,
                MONO27B_N_ROT_DIMS); TRACE("mk");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "q_rope", qb, MONO27B_TARGET_Q_DIM, 64);
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "k_rope", kb, MONO27B_TARGET_KV_DIM, 64);
            }

            for (int hh = 0; hh < MONO27B_TARGET_N_KV_HEAD; ++hh) {
                size_t off = ((size_t)pos * MONO27B_TARGET_N_KV_HEAD + hh) * MONO27B_TARGET_HEAD_DIM;
                cudaMemcpyAsync(st->kv_cache_k[fa_i] + off, kb + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(st->kv_cache_v[fa_i] + off,
                    kb + MONO27B_TARGET_KV_DIM + hh * MONO27B_TARGET_HEAD_DIM,
                    MONO27B_TARGET_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            TRACE("kvc");

            // Attention: parallel cooperative kernel
            {
                int kvl = pos + 1;
                // Use max_ctx shared memory always for graph capture compatibility
                int smem_kvl = max_ctx > 0 ? max_ctx : kvl;
                size_t smem_sz = ((size_t)smem_kvl + 2 + MONO27B_TARGET_HEAD_DIM) * sizeof(float);
                k_attn_parallel<<<MONO27B_TARGET_N_HEAD, MONO27B_TARGET_HEAD_DIM, smem_sz>>>(
                    qb, st->kv_cache_k[fa_i], st->kv_cache_v[fa_i], qb,
                    kvl, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_N_KV_HEAD,
                    MONO27B_TARGET_HEAD_DIM, max_ctx, 1.0f / sqrtf(MONO27B_TARGET_HEAD_DIM));
            }
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_raw", qb, MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM);
            }
            // Gate: sigmoid(gate_q) * attn_output
            k_elem_sigmoid_mul<<<(MONO27B_TARGET_Q_DIM + 255) / 256, 256>>>(
                qb, qb + MONO27B_TARGET_Q_DIM, qb, MONO27B_TARGET_Q_DIM); TRACE("gt");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_gated", qb, MONO27B_TARGET_Q_DIM, MONO27B_TARGET_Q_DIM);
            }

            MV(L.wo, qb, h2); TRACE("wo");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "attn_out", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("res1");

            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN); TRACE("porm");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "post_norm", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            MV(L.ffn_gate, h, fb); TRACE("fg");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "ffn_gate", fb, MONO27B_TARGET_FFN, MONO27B_TARGET_FFN);
            }
            MV(L.ffn_up, h, kb); TRACE("fu");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "ffn_up", kb, MONO27B_TARGET_FFN, MONO27B_TARGET_FFN);
            }
            k_elem_swiglu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN); TRACE("mul");
            MV(L.ffn_down, fb, h); TRACE("fd");
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "ffn_down", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN);
            CHECK_FINITE_FMT("attn layer %d output", il, h, MONO27B_TARGET_HIDDEN);
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "attn", il, pos, tok, "post_ffn", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
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
            if (dump_step && il == 0) {
                // Dump FULL wqkv_gate (6144) for Python comparison
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "wqkv_gate", gb, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
            }
            if (dump_step && il < 3) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "attn_norm", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "linear_attn_qkv_mixed", sb, MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_CH);
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "z", gb, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
                // Dump FULL attn_norm (h2) for verification (20KB)
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "wqkv", sb, MONO27B_SSM_CONV_CH, 16);
            }

            if (L.ssm_beta.ptr && L.ssm_alpha.ptr && L.ssm_dt_bias.ptr && L.ssm_a_log.ptr) {
                int dr = MONO27B_SSM_DT_RANK;
                MV(L.ssm_beta, h2, kb);
                k_elem_sigmoid<<<(dr + 31) / 32, 32>>>(kb, dr);
                CHECK_FINITE("ssm beta", kb, dr);
                MV(L.ssm_alpha, h2, qb);
                k_elem_softplus<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_dt_bias), qb, dr);
                k_elem_mul<<<(dr + 31) / 32, 32>>>(qb, WV(L.ssm_a_log), dr);
                CHECK_FINITE("ssm dt", qb, dr);
                if (dump_step && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "beta", kb, dr);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "dt", qb, dr);
                }

            // Conv1D
            if (dump_step && il == 0) {
                // Debug dump first 8 conv weights
                std::vector<float> conv_w_host(MONO27B_SSM_CONV_KERN * 2); // 8 weights for first 2 channels
                if (L.ssm_conv1d.ggml_type == MONO27B_GGML_TYPE_F16) {
                    std::vector<__half> host_half(MONO27B_SSM_CONV_KERN * 2);
                    cudaMemcpy(host_half.data(), L.ssm_conv1d.ptr, sizeof(__half) * MONO27B_SSM_CONV_KERN * 2, cudaMemcpyDeviceToHost);
                    for (int i = 0; i < MONO27B_SSM_CONV_KERN * 2; i++) conv_w_host[i] = __half2float(host_half[i]);
                } else {
                    cudaMemcpy(conv_w_host.data(), L.ssm_conv1d.ptr, sizeof(float) * MONO27B_SSM_CONV_KERN * 2, cudaMemcpyDeviceToHost);
                }
                std::fprintf(debug_fp, "dbg\t0\t%d\t%d\tconv_w\t%d\t", pos, tok, MONO27B_SSM_CONV_KERN * 2);
                for (int i = 0; i < MONO27B_SSM_CONV_KERN * 2; i++) {
                    std::fprintf(debug_fp, "%s%.9g", i ? "," : "", conv_w_host[i]);
                }
                std::fprintf(debug_fp, "\n");
                // Also dump wqkv values that feed into conv
                std::vector<float> inp_host(MONO27B_SSM_CONV_CH);
                cudaMemcpy(inp_host.data(), sb, sizeof(float) * MONO27B_SSM_CONV_CH, cudaMemcpyDeviceToHost);
                std::fprintf(debug_fp, "dbg\t0\t%d\t%d\tconv_inp\t%d\t", pos, tok, MONO27B_SSM_CONV_CH);
                for (int i = 0; i < MONO27B_SSM_CONV_CH; i++) {
                    std::fprintf(debug_fp, "%s%.9g", i ? "," : "", inp_host[i]);
                }
                std::fprintf(debug_fp, "\n");
            }
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
                if (dump_step && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "conv_raw", sb, MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_CH);
                }
                k_elem_silu<<<(MONO27B_SSM_CONV_CH + 255) / 256, 256>>>(sb, MONO27B_SSM_CONV_CH);
                CHECK_FINITE("ssm conv", sb, MONO27B_SSM_CONV_CH);
                if (dump_step && il < 3) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "conv", sb, MONO27B_SSM_CONV_CH, MONO27B_SSM_CONV_CH);
                }

                float * qr = sb;
                float * kr = sb + MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;
                float * vr = sb + 2 * MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K;

                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(qr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP);
                k_l2_norm_g<<<MONO27B_SSM_N_GROUP, 128>>>(kr, MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP);
                CHECK_FINITE("ssm qk", sb, MONO27B_SSM_CONV_CH);

                // Gated DeltaNet
                if (st->ssm_state[ssm_i]) {
                    k_deltanet_ggml_launch(qr, kr, vr, qb, kb,
                        st->ssm_state[ssm_i], gb, st->ssm_state[ssm_i],
                        MONO27B_SSM_N_GROUP, MONO27B_SSM_DT_RANK,
                        MONO27B_SSM_HEAD_V, MONO27B_SSM_HEAD_K);
                    CHECK_FINITE("ssm deltanet", gb, MONO27B_SSM_D_INNER);
                    if (dump_step && il < 4) {
                        debug_dump_vec(debug_fp, "ssm", il, pos, tok, "deltanet", gb, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
                    }
                }

                // Gate: wqkv_gate @ h2 → siLU → mul with rms_norm(ssm_out)
                MV(L.wqkv_gate, h2, fb); TRACE("re-g");
                k_elem_silu<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(fb, MONO27B_SSM_D_INNER); TRACE("g_silu");
                // Use kb (17408 floats) as intermediate for gated norm output;
                // h2 is only 5120 floats but the gated norm output is 6144 floats.
                for (int r = 0; r < MONO27B_SSM_DT_RANK; ++r) {
                    const float * w_norm = nullptr;
                    if (L.ssm_norm.ptr) {
                        w_norm = WV(L.ssm_norm);
                    }
                    k_rms_norm_mulw<256><<<1, 256>>>(
                        gb + (size_t)r * MONO27B_SSM_HEAD_V,
                        w_norm,
                        kb + (size_t)r * MONO27B_SSM_HEAD_V,
                        MONO27B_SSM_HEAD_V,
                        MONO27B_RMS_EPS);
                }
                TRACE("grms");
                k_elem_mul<<<(MONO27B_SSM_D_INNER + 255) / 256, 256>>>(kb, fb, MONO27B_SSM_D_INNER); TRACE("gmul2");
                if (dump_step && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "final_output", kb, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "q_conv_predelta", qr, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "k_conv_predelta", kr, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K, MONO27B_SSM_N_GROUP * MONO27B_SSM_HEAD_K);
                }
                MV(L.ssm_out, kb, sb); TRACE("ssmo");
                if (dump_step && il < 4) {
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "gate", fb, MONO27B_SSM_D_INNER);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "rms_gated", h2, MONO27B_SSM_D_INNER, MONO27B_SSM_D_INNER);
                    debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ssm_out", sb, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
                }
                k_elem_copy<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, sb, MONO27B_TARGET_HIDDEN); TRACE("scp");
                k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h2, h, MONO27B_TARGET_HIDDEN); TRACE("sadd");
                CHECK_FINITE_FMT("ssm layer %d output", il, h2, MONO27B_TARGET_HIDDEN);
                if (dump_step && il < MONO27B_TARGET_LAYERS) {
                    // layer_out will be dumped after FFN below
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
                CHECK_FINITE_FMT("ssm layer %d output", il, h2, MONO27B_TARGET_HIDDEN);
            }

        ssm_ffn:
            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN);
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "post_norm", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            MV(L.ffn_gate, h, fb);
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_gate", fb, MONO27B_TARGET_FFN, MONO27B_TARGET_FFN);
            }
            MV(L.ffn_up, h, kb);
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_up", kb, MONO27B_TARGET_FFN, MONO27B_TARGET_FFN);
            }
            k_elem_swiglu<<<(MONO27B_TARGET_FFN + 255) / 256, 256>>>(fb, kb, MONO27B_TARGET_FFN);
            if (dump_step && il < 4) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_mul", fb, MONO27B_TARGET_FFN, MONO27B_TARGET_FFN);
            }
            MV(L.ffn_down, fb, h);
            if (dump_step && il < 3) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "ffn_down", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
            }
            k_elem_add<<<(MONO27B_TARGET_HIDDEN + 255) / 256, 256>>>(h, h2, MONO27B_TARGET_HIDDEN);
            if (dump_step && il < 3) {
                debug_dump_vec(debug_fp, "ssm", il, pos, tok, "layer_out", h, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
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
    if (debug_fp) {
        cudaPointerAttributes attrs{};
        cudaError_t pa = cudaPointerGetAttributes(&attrs, h2);
        std::fprintf(debug_fp, "out\t%d\t%d\t%d\t%s\tptr\th2\t%p\t%s\n",
                     -1, pos, tok, "output_norm",
                     (void *)h2, pa == cudaSuccess ? "device" : cudaGetErrorString(pa));
        if (dump_step) {
            debug_dump_vec(debug_fp, "out", -1, pos, tok, "output_norm", h2, MONO27B_TARGET_HIDDEN, MONO27B_TARGET_HIDDEN);
        }
    }
    // REAL LM head: quantize h2 to Q8_1 once, then use dp4a for all vocab rows.
    // This matches llama.cpp's mmvq (matrix-vector quantized) path for Q6_K.
    {
        int total = (int)we->lm_head.row_count;
        int rb = (int)we->lm_head.row_blocks;
        auto * base = (const BlockQ6K *)we->lm_head.ptr;
        // n_q8 = rb * 8 = 20 * 8 = 160, well within scratch capacity of 544
        int n_q8 = rb * 8;
        if (g_q8_scratch && n_q8 <= 544) {
            // Quantize h2 → Q8_1 once (same input for all rows)
            k_quant_q8_1<<<(n_q8 + 127) / 128, 128>>>(h2, g_q8_scratch, MONO27B_TARGET_HIDDEN);
            sync_err = cudaDeviceSynchronize();
            if (sync_err == cudaSuccess) {
                // One big kernel launch for all vocab rows
                k_q6k_mv_q8_dp4a<<<total, 128>>>(base, g_q8_scratch, out->logits, rb, total);
                sync_err = cudaDeviceSynchronize();
            }
        }
        if (!g_q8_scratch || n_q8 > 544 || sync_err != cudaSuccess) {
            // Fallback: F32 dequant path in chunks
            sync_err = cudaSuccess;
            int chunk = 4096;
            for (int off = 0; off < total; off += chunk) {
                int n = (off + chunk > total) ? total - off : chunk;
                k_q6k_mt<<<n, 256>>>(base + (size_t)off * rb, h2, out->logits + off, rb, n);
                sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) break;
            }
        }
        if (sync_err != cudaSuccess) {
            // Last-resort fallback
            cudaMemset(out->logits, 0, MONO27B_TARGET_VOCAB * sizeof(float));
            float v = 1.0f;
            cudaMemcpy(out->logits, &v, 4, cudaMemcpyHostToDevice);
            sync_err = cudaDeviceSynchronize();
        }
        if (dump_step) {
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
        k_q4k_embed<<<1, 256>>>(
            (const BlockQ4K *)we->tok_embd.ptr,
            token_id,
            hidden,
            MONO27B_TARGET_HIDDEN,
            (int)we->tok_embd.row_blocks);
        cudaError_t e = cudaDeviceSynchronize();
        if (e == cudaSuccess) {
            return true;
        }
        // Keep the old host-side path as a fallback so tokenization still works
        // if the device launch fails on a particular runtime.
        if (q4k_embed_host((const BlockQ4K *)we->tok_embd.ptr, token_id, hidden,
                           MONO27B_TARGET_HIDDEN, (int)we->tok_embd.row_blocks)) {
            return true;
        }
        std::snprintf(error, error_cap, "q4k embed failed: %s", cudaGetErrorString(e));
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
