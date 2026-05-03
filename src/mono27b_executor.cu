#include "mono27b_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// ─── Runtime Layout ─────────────────────────────────────────────────────────

struct KvArenaDesc {
    size_t bytes_per_token = 0;
    size_t total_bytes = 0;
};

struct RollbackArenaDesc {
    size_t bytes_per_layer = 0;
    size_t total_bytes = 0;
};

struct WorkspaceDesc {
    size_t draft_logits_bytes = 0;
    size_t hidden_bytes = 0;
    size_t total_bytes = 0;
};

struct MatrixRunDesc {
    uint32_t row_elems = 0;
    uint32_t row_count = 0;
    uint32_t rows_to_run = 0;
};

struct VectorStats {
    float mean_abs = 0.0f;
    float l2 = 0.0f;
    float max_abs = 0.0f;
};

struct ArgmaxResult {
    int index = -1;
    float value = 0.0f;
};

struct PromptState {
    std::vector<float> input;
    std::vector<float> normed;
};

// ─── Quant Layouts ──────────────────────────────────────────────────────────

struct Q4KMTag {};

struct BlockQ40K {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ80 {
    __half d;
    int8_t qs[32];
};

struct BlockQ60K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <typename QuantTag>
struct QuantReader;

template <>
struct QuantReader<Q4KMTag> {
    __device__ static float unpack_first(const unsigned char * src) {
        const BlockQ40K * block = reinterpret_cast<const BlockQ40K *>(src);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(0, block->scales, sc, m);
        const float d = __half2float(block->d) * static_cast<float>(sc);
        const float min = __half2float(block->dmin) * static_cast<float>(m);
        const float q = static_cast<float>(block->qs[0] & 0x0F);
        return d * q - min;
    }
};

// ─── Error Helpers ──────────────────────────────────────────────────────────

static void set_error(char * error, size_t error_cap, const char * message) {
    if (!error || error_cap == 0) {
        return;
    }
    std::snprintf(error, error_cap, "%s", message);
}

static void set_cuda_error(char * error,
                           size_t error_cap,
                           const char * prefix,
                           cudaError_t err) {
    if (!error || error_cap == 0) {
        return;
    }
    std::snprintf(error, error_cap, "%s: %s", prefix, cudaGetErrorString(err));
}

struct DeviceBuffer {
    void * ptr = nullptr;
    size_t bytes = 0;

    ~DeviceBuffer() {
        reset();
    }

    void reset() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }

    bool alloc(size_t nbytes, char * error, size_t error_cap, const char * label) {
        reset();
        if (nbytes == 0) {
            return true;
        }
        const cudaError_t err = cudaMalloc(&ptr, nbytes);
        if (err != cudaSuccess) {
            set_cuda_error(error, error_cap, label, err);
            return false;
        }
        bytes = nbytes;
        return true;
    }

    template <typename T>
    T * as() {
        return reinterpret_cast<T *>(ptr);
    }

    template <typename T>
    const T * as() const {
        return reinterpret_cast<const T *>(ptr);
    }
};

// ─── Host-Side Prompt Synthesis ─────────────────────────────────────────────

static std::vector<float> make_prompt_vector(const char * prompt, uint32_t n) {
    std::vector<float> x(n, 0.0f);
    if (!prompt || n == 0) {
        return x;
    }
    const size_t prompt_len = std::strlen(prompt);
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char ch = prompt_len > 0 ? static_cast<unsigned char>(prompt[i % prompt_len]) : 0U;
        const float centered = (static_cast<int>(ch) - 96) / 32.0f;
        const float harmonic = 0.03125f * static_cast<float>(i % 7U);
        const float phase = 0.015625f * static_cast<float>((i / 17U) % 5U);
        x[i] = centered + harmonic - phase;
    }
    return x;
}

static std::vector<float> make_prompt_token_vector(const int32_t * prompt_ids,
                                                   size_t prompt_id_count,
                                                   uint32_t n) {
    std::vector<float> x(n, 0.0f);
    if (!prompt_ids || prompt_id_count == 0 || n == 0) {
        return x;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t tok = prompt_ids[i % prompt_id_count];
        const float base = static_cast<float>((tok % 257) - 128) / 48.0f;
        const float harmonic = 0.0234375f * static_cast<float>(i % 11U);
        const float phase = 0.0078125f * static_cast<float>((tok / 17 + static_cast<int32_t>(i)) % 13);
        x[i] = base + harmonic - phase;
    }
    return x;
}

static MatrixRunDesc make_run_desc(uint32_t row_elems,
                                   uint32_t row_count,
                                   uint32_t max_rows) {
    MatrixRunDesc out;
    out.row_elems = row_elems;
    out.row_count = row_count;
    out.rows_to_run = std::min(row_count, max_rows);
    return out;
}

// ─── Primitive CUDA Kernels ─────────────────────────────────────────────────

template <typename QuantTag>
__global__ static void k_quant_probe_first(const unsigned char * src,
                                           float * dst) {
    if (threadIdx.x == 0) {
        dst[0] = QuantReader<QuantTag>::unpack_first(src);
    }
}

__global__ static void k_rms_norm(const float * x,
                                  float * y,
                                  int n,
                                  float eps) {
    __shared__ float shared[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = x[i] * inv_rms;
    }
}

__global__ static void k_vector_stats(const float * x,
                                      int n,
                                      float * out_abs_sum,
                                      float * out_sq_sum,
                                      float * out_max_abs) {
    __shared__ float sh_abs[256];
    __shared__ float sh_sq[256];
    __shared__ float sh_max[256];
    float abs_sum = 0.0f;
    float sq_sum = 0.0f;
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        const float av = fabsf(v);
        abs_sum += av;
        sq_sum += v * v;
        max_abs = fmaxf(max_abs, av);
    }
    sh_abs[threadIdx.x] = abs_sum;
    sh_sq[threadIdx.x] = sq_sum;
    sh_max[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_abs[threadIdx.x] += sh_abs[threadIdx.x + stride];
            sh_sq[threadIdx.x] += sh_sq[threadIdx.x + stride];
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sh_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_abs_sum[0] = sh_abs[0];
        out_sq_sum[0] = sh_sq[0];
        out_max_abs[0] = sh_max[0];
    }
}

__global__ static void k_argmax(const float * x,
                                int n,
                                int * out_index,
                                float * out_value) {
    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    float best_val = -1.0e30f;
    int best_idx = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    sh_val[threadIdx.x] = best_val;
    sh_idx[threadIdx.x] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && sh_val[threadIdx.x + stride] > sh_val[threadIdx.x]) {
            sh_val[threadIdx.x] = sh_val[threadIdx.x + stride];
            sh_idx[threadIdx.x] = sh_idx[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_index[0] = sh_idx[0];
        out_value[0] = sh_val[0];
    }
}

__global__ static void k_q80_matvec_rows(const BlockQ80 * weights,
                                         const float * x,
                                         float * y,
                                         int row_blocks,
                                         int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    __shared__ float shared[128];
    float sum = 0.0f;
    const BlockQ80 * row_ptr = weights + static_cast<size_t>(row) * row_blocks;
    for (int block = threadIdx.x; block < row_blocks; block += blockDim.x) {
        const BlockQ80 & qb = row_ptr[block];
        const float d = __half2float(qb.d);
        const int base = block * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum += d * static_cast<float>(qb.qs[i]) * x[base + i];
        }
    }

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = shared[0];
    }
}

__global__ static void k_q4k_matvec_rows(const BlockQ40K * weights,
                                         const float * x,
                                         float * y,
                                         int row_blocks,
                                         int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    __shared__ float shared[128];
    float sum = 0.0f;
    const BlockQ40K * row_ptr = weights + static_cast<size_t>(row) * row_blocks;
    for (int block = threadIdx.x; block < row_blocks; block += blockDim.x) {
        const BlockQ40K & qb = row_ptr[block];
        const float d = __half2float(qb.d);
        const float min = __half2float(qb.dmin);
        for (int group = 0; group < 4; ++group) {
            uint8_t sc0 = 0;
            uint8_t m0 = 0;
            uint8_t sc1 = 0;
            uint8_t m1 = 0;
            get_scale_min_k4(group * 2 + 0, qb.scales, sc0, m0);
            get_scale_min_k4(group * 2 + 1, qb.scales, sc1, m1);
            const float d0 = d * static_cast<float>(sc0);
            const float d1 = d * static_cast<float>(sc1);
            const float min0 = min * static_cast<float>(m0);
            const float min1 = min * static_cast<float>(m1);
            const int base = block * 256 + group * 64;
            const uint8_t * qg = qb.qs + group * 32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                sum += (d0 * static_cast<float>(qg[i] & 0x0F) - min0) * x[base + i];
                sum += (d1 * static_cast<float>(qg[i] >> 4) - min1) * x[base + 32 + i];
            }
        }
    }

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = shared[0];
    }
}

__global__ static void k_q4k_gather_rows(const BlockQ40K * weights,
                                         const int32_t * ids,
                                         int id_count,
                                         float * out,
                                         int row_elems,
                                         int row_blocks,
                                         int row_count) {
    const int row_idx = blockIdx.x;
    if (row_idx >= id_count) {
        return;
    }
    const int32_t token_id = ids[row_idx];
    if (token_id < 0 || token_id >= row_count) {
        return;
    }
    const int token = token_id;
    const BlockQ40K * row_ptr = weights + static_cast<size_t>(token) * row_blocks;
    float * row_out = out + static_cast<size_t>(row_idx) * row_elems;
    for (int elem = threadIdx.x; elem < row_elems; elem += blockDim.x) {
        const int block = elem / 256;
        const int intra = elem % 256;
        const int group = intra / 64;
        const int within = intra % 64;
        const BlockQ40K & qb = row_ptr[block];
        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        get_scale_min_k4(group * 2 + 0, qb.scales, sc0, m0);
        get_scale_min_k4(group * 2 + 1, qb.scales, sc1, m1);
        const float d0 = __half2float(qb.d) * static_cast<float>(sc0);
        const float d1 = __half2float(qb.d) * static_cast<float>(sc1);
        const float min0 = __half2float(qb.d) * 0.0f + __half2float(qb.dmin) * static_cast<float>(m0);
        const float min1 = __half2float(qb.d) * 0.0f + __half2float(qb.dmin) * static_cast<float>(m1);
        const uint8_t * qg = qb.qs + group * 32;
        if (within < 32) {
            row_out[elem] = d0 * static_cast<float>(qg[within] & 0x0F) - min0;
        } else {
            row_out[elem] = d1 * static_cast<float>(qg[within - 32] >> 4) - min1;
        }
    }
}

__global__ static void k_q6k_matvec_rows(const BlockQ60K * weights,
                                         const float * x,
                                         float * y,
                                         int row_blocks,
                                         int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    __shared__ float shared[128];
    float sum = 0.0f;
    const BlockQ60K * row_ptr = weights + static_cast<size_t>(row) * row_blocks;
    for (int block = threadIdx.x; block < row_blocks; block += blockDim.x) {
        const BlockQ60K & qb = row_ptr[block];
        const float d = __half2float(qb.d);
        for (int lane = 0; lane < 32; ++lane) {
            const int ip = lane / 16;
            const int il = lane % 16;
            const int is = 8 * ip + il / 8;
            const uint8_t * ql = qb.ql + 64 * ip + il;
            const uint8_t qh = qb.qh[32 * ip + il];
            const int8_t * sc = qb.scales + is;
            const int base = block * 256 + 128 * ip + il;
            const float s0 = d * static_cast<float>(sc[0]);
            const float s1 = d * static_cast<float>(sc[2]);
            const float s2 = d * static_cast<float>(sc[4]);
            const float s3 = d * static_cast<float>(sc[6]);
            const float q0 = static_cast<float>(static_cast<int>(((ql[0] & 0x0F) | (((qh >> 0) & 3) << 4))) - 32);
            const float q1 = static_cast<float>(static_cast<int>(((ql[32] & 0x0F) | (((qh >> 2) & 3) << 4))) - 32);
            const float q2 = static_cast<float>(static_cast<int>(((ql[0] >> 4) | (((qh >> 4) & 3) << 4))) - 32);
            const float q3 = static_cast<float>(static_cast<int>(((ql[32] >> 4) | (((qh >> 6) & 3) << 4))) - 32);
            sum += s0 * q0 * x[base + 0];
            sum += s1 * q1 * x[base + 32];
            sum += s2 * q2 * x[base + 64];
            sum += s3 * q3 * x[base + 96];
        }
    }

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = shared[0];
    }
}

// ─── Runtime Layout Helpers ─────────────────────────────────────────────────

static KvArenaDesc make_kv_desc(size_t ctx) {
    KvArenaDesc out;
    out.bytes_per_token = static_cast<size_t>(MONO27B_TARGET_FA_LAYERS) *
                          static_cast<size_t>(MONO27B_TARGET_N_KV_HEAD) *
                          static_cast<size_t>(MONO27B_TARGET_HEAD_DIM) * 2U;
    out.total_bytes = out.bytes_per_token * ctx;
    return out;
}

static RollbackArenaDesc make_rollback_desc() {
    RollbackArenaDesc out;
    out.bytes_per_layer = 1024U;
    out.total_bytes = out.bytes_per_layer * static_cast<size_t>(MONO27B_TARGET_SSM_LAYERS);
    return out;
}

static WorkspaceDesc make_workspace_desc() {
    WorkspaceDesc out;
    out.draft_logits_bytes = static_cast<size_t>(MONO27B_DRAFT_BLOCK) * sizeof(float) * 1024U;
    out.hidden_bytes = static_cast<size_t>(MONO27B_DRAFT_BLOCK) *
                       static_cast<size_t>(MONO27B_TARGET_HIDDEN) * sizeof(float);
    out.total_bytes = out.draft_logits_bytes + out.hidden_bytes;
    return out;
}

// ─── Host / CUDA Bridge Helpers ─────────────────────────────────────────────

template <typename QuantTag>
static bool run_quant_probe(const unsigned char * src_host,
                            size_t nbytes,
                            float & first_value,
                            char * error,
                            size_t error_cap) {
    DeviceBuffer src_dev;
    DeviceBuffer dst_dev;
    if (!src_dev.alloc(nbytes, error, error_cap, "cuda alloc quant src") ||
        !dst_dev.alloc(sizeof(float), error, error_cap, "cuda alloc quant dst")) {
        return false;
    }
    cudaError_t err = cudaMemcpy(src_dev.ptr, src_host, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload quant probe", err);
        return false;
    }
    k_quant_probe_first<QuantTag><<<1, 32>>>(src_dev.as<unsigned char>(), dst_dev.as<float>());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda quant probe kernel", err);
        return false;
    }
    err = cudaMemcpy(&first_value, dst_dev.ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download quant probe", err);
        return false;
    }
    return true;
}

static bool run_rms_norm(const std::vector<float> & input,
                         std::vector<float> & output,
                         char * error,
                         size_t error_cap) {
    if (input.empty()) {
        set_error(error, error_cap, "rms norm received empty input");
        return false;
    }
    output.resize(input.size());
    DeviceBuffer x_dev;
    DeviceBuffer y_dev;
    if (!x_dev.alloc(input.size() * sizeof(float), error, error_cap, "cuda alloc rms x") ||
        !y_dev.alloc(output.size() * sizeof(float), error, error_cap, "cuda alloc rms y")) {
        return false;
    }
    cudaError_t err = cudaMemcpy(x_dev.ptr, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload rms input", err);
        return false;
    }
    k_rms_norm<<<1, 256>>>(x_dev.as<float>(), y_dev.as<float>(), static_cast<int>(input.size()), 1e-5f);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda rms kernel", err);
        return false;
    }
    err = cudaMemcpy(output.data(), y_dev.ptr, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download rms output", err);
        return false;
    }
    return true;
}

template <typename BlockT>
static bool run_matrix_kernel(const unsigned char * weights_host,
                              size_t weights_bytes,
                              const std::vector<float> & x_host,
                              uint32_t row_count,
                              int row_blocks,
                              void (*launch)(const BlockT *, const float *, float *, int, int),
                              std::vector<float> & output,
                              char * error,
                              size_t error_cap) {
    output.assign(row_count, 0.0f);
    DeviceBuffer weights_dev;
    DeviceBuffer x_dev;
    DeviceBuffer y_dev;
    if (!weights_dev.alloc(weights_bytes, error, error_cap, "cuda alloc weights") ||
        !x_dev.alloc(x_host.size() * sizeof(float), error, error_cap, "cuda alloc x") ||
        !y_dev.alloc(output.size() * sizeof(float), error, error_cap, "cuda alloc y")) {
        return false;
    }
    cudaError_t err = cudaMemcpy(weights_dev.ptr, weights_host, weights_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload weights", err);
        return false;
    }
    err = cudaMemcpy(x_dev.ptr, x_host.data(), x_host.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload x", err);
        return false;
    }
    launch(weights_dev.as<BlockT>(), x_dev.as<float>(), y_dev.as<float>(), row_blocks, static_cast<int>(row_count));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda matvec kernel", err);
        return false;
    }
    err = cudaMemcpy(output.data(), y_dev.ptr, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download matvec output", err);
        return false;
    }
    return true;
}

static void launch_q80_matvec(const BlockQ80 * weights,
                              const float * x,
                              float * y,
                              int row_blocks,
                              int row_count) {
    k_q80_matvec_rows<<<row_count, 128>>>(weights, x, y, row_blocks, row_count);
}

static void launch_q4k_matvec(const BlockQ40K * weights,
                              const float * x,
                              float * y,
                              int row_blocks,
                              int row_count) {
    k_q4k_matvec_rows<<<row_count, 128>>>(weights, x, y, row_blocks, row_count);
}

static void launch_q4k_gather(const BlockQ40K * weights,
                              const int32_t * ids,
                              int id_count,
                              float * out,
                              int row_elems,
                              int row_blocks,
                              int row_count) {
    k_q4k_gather_rows<<<id_count, 128>>>(weights, ids, id_count, out, row_elems, row_blocks, row_count);
}

static void launch_q6k_matvec(const BlockQ60K * weights,
                              const float * x,
                              float * y,
                              int row_blocks,
                              int row_count) {
    k_q6k_matvec_rows<<<row_count, 128>>>(weights, x, y, row_blocks, row_count);
}

static bool run_q80_matvec(const unsigned char * weights_host,
                           size_t weights_bytes,
                           const MatrixRunDesc & desc,
                           const std::vector<float> & x_host,
                           std::vector<float> & output,
                           char * error,
                           size_t error_cap) {
    if (!weights_host || desc.row_elems == 0 || desc.rows_to_run == 0 || (desc.row_elems % 32U) != 0U) {
        set_error(error, error_cap, "invalid q8_0 matrix metadata");
        return false;
    }
    const size_t row_bytes = (static_cast<size_t>(desc.row_elems) / 32U) * sizeof(BlockQ80);
    if (weights_bytes < row_bytes * static_cast<size_t>(desc.rows_to_run)) {
        set_error(error, error_cap, "q8_0 matrix blob too small");
        return false;
    }
    return run_matrix_kernel<BlockQ80>(weights_host,
                                       weights_bytes,
                                       x_host,
                                       desc.rows_to_run,
                                       static_cast<int>(desc.row_elems / 32U),
                                       launch_q80_matvec,
                                       output,
                                       error,
                                       error_cap);
}

static bool run_q4k_matvec(const unsigned char * weights_host,
                           size_t weights_bytes,
                           const MatrixRunDesc & desc,
                           const std::vector<float> & x_host,
                           std::vector<float> & output,
                           char * error,
                           size_t error_cap) {
    if (!weights_host || desc.row_elems == 0 || desc.rows_to_run == 0 || (desc.row_elems % 256U) != 0U) {
        set_error(error, error_cap, "invalid q4_k matrix metadata");
        return false;
    }
    const size_t row_bytes = (static_cast<size_t>(desc.row_elems) / 256U) * sizeof(BlockQ40K);
    if (weights_bytes < row_bytes * static_cast<size_t>(desc.rows_to_run)) {
        set_error(error, error_cap, "q4_k matrix blob too small");
        return false;
    }
    return run_matrix_kernel<BlockQ40K>(weights_host,
                                        weights_bytes,
                                        x_host,
                                        desc.rows_to_run,
                                        static_cast<int>(desc.row_elems / 256U),
                                        launch_q4k_matvec,
                                        output,
                                        error,
                                        error_cap);
}

static bool run_q4k_gather(const unsigned char * weights_host,
                           size_t weights_bytes,
                           uint32_t row_elems,
                           const int32_t * ids_host,
                           size_t id_count,
                           std::vector<float> & output,
                           char * error,
                           size_t error_cap) {
    if (!weights_host || ids_host == nullptr || id_count == 0 || row_elems == 0 || (row_elems % 256U) != 0U) {
        set_error(error, error_cap, "invalid q4_k gather metadata");
        return false;
    }
    const int row_blocks = static_cast<int>(row_elems / 256U);
    const size_t row_bytes = static_cast<size_t>(row_blocks) * sizeof(BlockQ40K);
    if (row_bytes == 0 || weights_bytes < row_bytes || (weights_bytes % row_bytes) != 0U) {
        set_error(error, error_cap, "q4_k gather blob too small");
        return false;
    }
    const int row_count = static_cast<int>(weights_bytes / row_bytes);
    output.assign(static_cast<size_t>(id_count) * static_cast<size_t>(row_elems), 0.0f);
    DeviceBuffer weights_dev;
    DeviceBuffer ids_dev;
    DeviceBuffer out_dev;
    if (!weights_dev.alloc(weights_bytes, error, error_cap, "cuda alloc q4k gather weights") ||
        !ids_dev.alloc(id_count * sizeof(int32_t), error, error_cap, "cuda alloc q4k gather ids") ||
        !out_dev.alloc(output.size() * sizeof(float), error, error_cap, "cuda alloc q4k gather out")) {
        return false;
    }
    cudaError_t err = cudaMemcpy(weights_dev.ptr, weights_host, weights_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload q4k gather weights", err);
        return false;
    }
    err = cudaMemcpy(ids_dev.ptr, ids_host, id_count * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload q4k gather ids", err);
        return false;
    }
    launch_q4k_gather(weights_dev.as<BlockQ40K>(), ids_dev.as<int32_t>(), static_cast<int>(id_count), out_dev.as<float>(), static_cast<int>(row_elems), row_blocks, row_count);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda q4k gather kernel", err);
        return false;
    }
    err = cudaMemcpy(output.data(), out_dev.ptr, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download q4k gather", err);
        return false;
    }
    return true;
}

static bool mean_rows(const std::vector<float> & rows,
                      size_t row_count,
                      size_t row_elems,
                      std::vector<float> & mean) {
    if (row_count == 0 || row_elems == 0 || rows.size() < row_count * row_elems) {
        return false;
    }
    mean.assign(row_elems, 0.0f);
    for (size_t r = 0; r < row_count; ++r) {
        const float * row = rows.data() + r * row_elems;
        for (size_t i = 0; i < row_elems; ++i) {
            mean[i] += row[i];
        }
    }
    const float inv = 1.0f / static_cast<float>(row_count);
    for (float & v : mean) {
        v *= inv;
    }
    return true;
}

static size_t collect_top_k(const std::vector<float> & scores,
                            size_t k,
                            int32_t * out_ids,
                            size_t out_cap) {
    if (!out_ids || out_cap == 0 || scores.empty()) {
        return 0;
    }
    const size_t n = std::min({k, out_cap, scores.size()});
    std::vector<size_t> order(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        order[i] = i;
    }
    std::partial_sort(order.begin(),
                      order.begin() + static_cast<std::ptrdiff_t>(n),
                      order.end(),
                      [&scores](size_t a, size_t b) {
                          return scores[a] > scores[b];
                      });
    for (size_t i = 0; i < n; ++i) {
        out_ids[i] = static_cast<int32_t>(order[i]);
    }
    return n;
}

static bool run_q6k_matvec(const unsigned char * weights_host,
                           size_t weights_bytes,
                           const MatrixRunDesc & desc,
                           const std::vector<float> & x_host,
                           std::vector<float> & output,
                           char * error,
                           size_t error_cap) {
    if (!weights_host || desc.row_elems == 0 || desc.rows_to_run == 0 || (desc.row_elems % 256U) != 0U) {
        set_error(error, error_cap, "invalid q6_k matrix metadata");
        return false;
    }
    const size_t row_bytes = (static_cast<size_t>(desc.row_elems) / 256U) * sizeof(BlockQ60K);
    if (weights_bytes < row_bytes * static_cast<size_t>(desc.rows_to_run)) {
        set_error(error, error_cap, "q6_k matrix blob too small");
        return false;
    }
    return run_matrix_kernel<BlockQ60K>(weights_host,
                                        weights_bytes,
                                        x_host,
                                        desc.rows_to_run,
                                        static_cast<int>(desc.row_elems / 256U),
                                        launch_q6k_matvec,
                                        output,
                                        error,
                                        error_cap);
}

static bool analyze_vector(const std::vector<float> & x,
                           VectorStats & stats,
                           ArgmaxResult & argmax,
                           char * error,
                           size_t error_cap) {
    if (x.empty()) {
        set_error(error, error_cap, "vector analysis received empty data");
        return false;
    }

    DeviceBuffer x_dev;
    DeviceBuffer abs_dev;
    DeviceBuffer sq_dev;
    DeviceBuffer max_dev;
    DeviceBuffer idx_dev;
    DeviceBuffer val_dev;
    if (!x_dev.alloc(x.size() * sizeof(float), error, error_cap, "cuda alloc stats x") ||
        !abs_dev.alloc(sizeof(float), error, error_cap, "cuda alloc abs sum") ||
        !sq_dev.alloc(sizeof(float), error, error_cap, "cuda alloc sq sum") ||
        !max_dev.alloc(sizeof(float), error, error_cap, "cuda alloc max abs") ||
        !idx_dev.alloc(sizeof(int), error, error_cap, "cuda alloc argmax idx") ||
        !val_dev.alloc(sizeof(float), error, error_cap, "cuda alloc argmax val")) {
        return false;
    }

    cudaError_t err = cudaMemcpy(x_dev.ptr, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda upload stats input", err);
        return false;
    }
    k_vector_stats<<<1, 256>>>(x_dev.as<float>(),
                               static_cast<int>(x.size()),
                               abs_dev.as<float>(),
                               sq_dev.as<float>(),
                               max_dev.as<float>());
    k_argmax<<<1, 256>>>(x_dev.as<float>(),
                         static_cast<int>(x.size()),
                         idx_dev.as<int>(),
                         val_dev.as<float>());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda vector analysis", err);
        return false;
    }

    float abs_sum = 0.0f;
    float sq_sum = 0.0f;
    err = cudaMemcpy(&abs_sum, abs_dev.ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download abs sum", err);
        return false;
    }
    err = cudaMemcpy(&sq_sum, sq_dev.ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download sq sum", err);
        return false;
    }
    err = cudaMemcpy(&stats.max_abs, max_dev.ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download max abs", err);
        return false;
    }
    err = cudaMemcpy(&argmax.index, idx_dev.ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download argmax index", err);
        return false;
    }
    err = cudaMemcpy(&argmax.value, val_dev.ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "cuda download argmax value", err);
        return false;
    }

    stats.mean_abs = abs_sum / static_cast<float>(x.size());
    stats.l2 = std::sqrt(sq_sum);
    return true;
}

// ─── GPU Pointwise Scale Kernel ─────────────────────────────────────────────

__global__ static void k_apply_scale(const float * scale, float * x, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] *= scale[i];
    }
}

// ─── GPU Mean-Pool Kernel ────────────────────────────────────────────────────

// Each block handles one output element (0..row_len-1), threads reduce over n_rows
__global__ static void k_mean_pool_rows(const float * rows,
                                        float * out,
                                        int n_rows,
                                        int row_len) {
    const int elem = blockIdx.x;
    __shared__ float sh[256];
    float sum = 0.0f;
    for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
        sum += rows[static_cast<size_t>(r) * static_cast<size_t>(row_len) + static_cast<size_t>(elem)];
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[elem] = sh[0] / static_cast<float>(n_rows);
    }
}

} // namespace

extern "C" bool mono27b_executor_load_weights(
    const unsigned char * target_host, size_t target_bytes,
    uint32_t target_row_elems, uint32_t target_row_count,
    const unsigned char * draft_host, size_t draft_bytes,
    uint32_t draft_row_elems, uint32_t draft_row_count,
    const unsigned char * head_host, size_t head_bytes,
    uint32_t head_row_elems, uint32_t head_row_count,
    Mono27BExecutorWeights * out,
    char * error, size_t error_cap) {
    if (!target_host || target_bytes < 2 || !draft_host || draft_bytes < 2 ||
        !head_host || head_bytes < 2 || !out) {
        set_error(error, error_cap, "load_weights: invalid arguments");
        return false;
    }
    *out = Mono27BExecutorWeights{};
    cudaError_t err;

    err = cudaMalloc(&out->target_gpu, target_bytes);
    if (err != cudaSuccess) { set_cuda_error(error, error_cap, "malloc target emb", err); return false; }
    out->target_bytes = target_bytes;
    out->target_row_elems = target_row_elems;
    out->target_row_count = target_row_count;
    err = cudaMemcpy(out->target_gpu, target_host, target_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "upload target emb", err);
        mono27b_executor_free_weights(out);
        return false;
    }

    err = cudaMalloc(&out->draft_gpu, draft_bytes);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "malloc draft", err);
        mono27b_executor_free_weights(out);
        return false;
    }
    out->draft_bytes = draft_bytes;
    out->draft_row_elems = draft_row_elems;
    out->draft_row_count = draft_row_count;
    err = cudaMemcpy(out->draft_gpu, draft_host, draft_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "upload draft", err);
        mono27b_executor_free_weights(out);
        return false;
    }

    err = cudaMalloc(&out->head_gpu, head_bytes);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "malloc head", err);
        mono27b_executor_free_weights(out);
        return false;
    }
    out->head_bytes = head_bytes;
    out->head_row_elems = head_row_elems;
    out->head_row_count = head_row_count;
    err = cudaMemcpy(out->head_gpu, head_host, head_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "upload head", err);
        mono27b_executor_free_weights(out);
        return false;
    }
    return true;
}

extern "C" void mono27b_executor_free_weights(Mono27BExecutorWeights * w) {
    if (!w) { return; }
    if (w->target_gpu)     { cudaFree(w->target_gpu);     w->target_gpu     = nullptr; }
    if (w->draft_gpu)      { cudaFree(w->draft_gpu);      w->draft_gpu      = nullptr; }
    if (w->head_gpu)       { cudaFree(w->head_gpu);       w->head_gpu       = nullptr; }
    if (w->norm_scale_gpu) { cudaFree(w->norm_scale_gpu); w->norm_scale_gpu = nullptr; }
    w->target_bytes    = 0;
    w->draft_bytes     = 0;
    w->head_bytes      = 0;
    w->norm_scale_elems = 0;
}

extern "C" bool mono27b_executor_set_norm_scale(
    Mono27BExecutorWeights * w,
    const float * scale_host, uint32_t elems,
    char * error, size_t error_cap) {
    if (!w || !scale_host || elems == 0) {
        set_error(error, error_cap, "set_norm_scale: invalid arguments");
        return false;
    }
    if (w->norm_scale_gpu) {
        cudaFree(w->norm_scale_gpu);
        w->norm_scale_gpu = nullptr;
        w->norm_scale_elems = 0;
    }
    const size_t nbytes = static_cast<size_t>(elems) * sizeof(float);
    const cudaError_t err = cudaMalloc(&w->norm_scale_gpu, nbytes);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "malloc norm_scale", err);
        return false;
    }
    const cudaError_t err2 = cudaMemcpy(w->norm_scale_gpu, scale_host, nbytes, cudaMemcpyHostToDevice);
    if (err2 != cudaSuccess) {
        cudaFree(w->norm_scale_gpu);
        w->norm_scale_gpu = nullptr;
        set_cuda_error(error, error_cap, "upload norm_scale", err2);
        return false;
    }
    w->norm_scale_elems = elems;
    return true;
}

extern "C" bool mono27b_executor_run_step(
    const Mono27BExecutorWeights * w,
    const int32_t * token_ids, size_t token_count,
    int32_t * generated_ids, size_t generated_cap,
    size_t * generated_count,
    char * diag, size_t diag_cap,
    char * error, size_t error_cap) {
    if (!w || !w->target_gpu || !w->head_gpu || !token_ids || token_count == 0) {
        set_error(error, error_cap, "run_step: invalid arguments");
        return false;
    }
    if (generated_count) { *generated_count = 0; }

    const int n_tok      = static_cast<int>(token_count);
    const int row_elems  = static_cast<int>(w->target_row_elems);
    const int row_count  = static_cast<int>(w->target_row_count);
    const int row_blocks = row_elems / 256;
    const int head_row_count  = static_cast<int>(w->head_row_count);
    const int head_row_blocks = static_cast<int>(w->head_row_elems) / 256;

    int32_t * ids_dev    = nullptr;
    float   * embeds_dev = nullptr;
    float   * hidden_dev = nullptr;
    float   * normed_dev = nullptr;
    float   * logits_dev = nullptr;

    const size_t embeds_bytes = static_cast<size_t>(n_tok) * static_cast<size_t>(row_elems) * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(row_elems) * sizeof(float);
    const size_t logits_bytes = static_cast<size_t>(head_row_count) * sizeof(float);

    auto cleanup = [&]() {
        if (ids_dev)    { cudaFree(ids_dev);    ids_dev    = nullptr; }
        if (embeds_dev) { cudaFree(embeds_dev); embeds_dev = nullptr; }
        if (hidden_dev) { cudaFree(hidden_dev); hidden_dev = nullptr; }
        if (normed_dev) { cudaFree(normed_dev); normed_dev = nullptr; }
        if (logits_dev) { cudaFree(logits_dev); logits_dev = nullptr; }
    };

    cudaError_t err;
    if ((err = cudaMalloc(&ids_dev,    static_cast<size_t>(n_tok) * sizeof(int32_t))) != cudaSuccess ||
        (err = cudaMalloc(&embeds_dev, embeds_bytes))  != cudaSuccess ||
        (err = cudaMalloc(&hidden_dev, hidden_bytes))  != cudaSuccess ||
        (err = cudaMalloc(&normed_dev, hidden_bytes))  != cudaSuccess ||
        (err = cudaMalloc(&logits_dev, logits_bytes))  != cudaSuccess) {
        set_cuda_error(error, error_cap, "run_step: GPU alloc", err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(ids_dev, token_ids, static_cast<size_t>(n_tok) * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "run_step: upload ids", err);
        cleanup();
        return false;
    }

    // Gather Q4K embeddings for each token → embeds_dev [n_tok × row_elems]
    k_q4k_gather_rows<<<n_tok, 128>>>(
        reinterpret_cast<const BlockQ40K *>(w->target_gpu),
        ids_dev, n_tok, embeds_dev, row_elems, row_blocks, row_count);

    // Mean-pool all token embeddings → hidden_dev [row_elems]
    k_mean_pool_rows<<<row_elems, 256>>>(embeds_dev, hidden_dev, n_tok, row_elems);

    // RMS-normalize → normed_dev [row_elems]
    k_rms_norm<<<1, 256>>>(hidden_dev, normed_dev, row_elems, 1e-5f);

    // Apply learnable scale weights if available (output_norm.weight)
    if (w->norm_scale_gpu && w->norm_scale_elems == static_cast<uint32_t>(row_elems)) {
        k_apply_scale<<<1, 256>>>(
            reinterpret_cast<const float *>(w->norm_scale_gpu), normed_dev, row_elems);
    }

    // Project through LM head (Q6K output.weight) → logits_dev [head_row_count]
    k_q6k_matvec_rows<<<head_row_count, 128>>>(
        reinterpret_cast<const BlockQ60K *>(w->head_gpu),
        normed_dev, logits_dev, head_row_blocks, head_row_count);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "run_step: sync", err);
        cleanup();
        return false;
    }

    // Download logits to host for top-K selection (~1 MB for 248K vocab)
    std::vector<float> logits_host(static_cast<size_t>(head_row_count));
    err = cudaMemcpy(logits_host.data(), logits_dev, logits_bytes, cudaMemcpyDeviceToHost);
    cleanup();
    if (err != cudaSuccess) {
        set_cuda_error(error, error_cap, "run_step: download logits", err);
        return false;
    }

    // Top-K selection
    if (generated_ids && generated_cap > 0) {
        const size_t produced = collect_top_k(logits_host, generated_cap, generated_ids, generated_cap);
        if (generated_count) { *generated_count = produced; }
    }

    // Diagnostic string
    if (diag && diag_cap > 0) {
        float max_logit = logits_host.empty() ? 0.0f :
            *std::max_element(logits_host.begin(), logits_host.end());
        float abs_sum = 0.0f;
        for (float v : logits_host) { abs_sum += std::abs(v); }
        const int top1_id = (generated_count && *generated_count > 0 && generated_ids)
            ? generated_ids[0] : -1;
        std::snprintf(diag, diag_cap,
                      "top1=%d max_logit=%.4f mean_abs=%.4f vocab=%u tok=%d",
                      top1_id, max_logit,
                      abs_sum / static_cast<float>(logits_host.size()),
                      static_cast<unsigned>(head_row_count), n_tok);
    }
    return true;
}

extern "C" bool mono27b_executor_init(const Mono27BBlobHeader * header,
                                      int max_ctx,
                                      Mono27BRuntimeLayout * layout,
                                      char * error,
                                      size_t error_cap) {
    if (!header || !layout) {
        set_error(error, error_cap, "executor init received null input");
        return false;
    }
    if (std::strncmp(header->target_quant, MONO27B_TARGET_QUANT, sizeof(header->target_quant)) != 0) {
        set_error(error, error_cap, "executor requires q4_k_m target quant");
        return false;
    }
    if (std::strncmp(header->draft_quant, MONO27B_DRAFT_QUANT, sizeof(header->draft_quant)) != 0) {
        set_error(error, error_cap, "executor requires q8_0 draft quant");
        return false;
    }

    const size_t ctx = static_cast<size_t>(max_ctx > 0 ? max_ctx : header->max_ctx_hint);
    const KvArenaDesc kv = make_kv_desc(ctx);
    const RollbackArenaDesc rollback = make_rollback_desc();
    const WorkspaceDesc workspace = make_workspace_desc();
    layout->kv_bytes = kv.total_bytes;
    layout->rollback_bytes = rollback.total_bytes;
    layout->workspace_bytes = workspace.total_bytes;
    layout->state_bytes = layout->kv_bytes + layout->rollback_bytes + layout->workspace_bytes;
    return true;
}

extern "C" bool mono27b_executor_run_prompt(const char * prompt,
                                            const int32_t * prompt_ids,
                                            size_t prompt_id_count,
                                            const unsigned char * target_tensor,
                                            size_t target_tensor_bytes,
                                            uint32_t target_row_elems,
                                            uint32_t target_row_count,
                                            const unsigned char * draft_tensor,
                                            size_t draft_tensor_bytes,
                                            uint32_t draft_row_elems,
                                            uint32_t draft_row_count,
                                            const unsigned char * output_head_tensor,
                                            size_t output_head_tensor_bytes,
                                            uint32_t output_head_row_elems,
                                            uint32_t output_head_row_count,
                                            int32_t * generated_ids,
                                            size_t generated_cap,
                                            size_t * generated_count,
                                            char * output,
                                            size_t output_cap,
                                            char * error,
                                            size_t error_cap) {
    if (!prompt || !output || output_cap == 0 || !target_tensor || !draft_tensor ||
        !output_head_tensor || target_tensor_bytes < 2 || draft_tensor_bytes < 2 || output_head_tensor_bytes < 2) {
        set_error(error, error_cap, "executor run received invalid buffers");
        return false;
    }
    if (generated_count) {
        *generated_count = 0;
    }

    float q4_probe = 0.0f;
    if (!run_quant_probe<Q4KMTag>(target_tensor, target_tensor_bytes, q4_probe, error, error_cap)) {
        return false;
    }

    const MatrixRunDesc target_desc = make_run_desc(target_row_elems, target_row_count, target_row_count);
    const MatrixRunDesc draft_desc = make_run_desc(draft_row_elems, draft_row_count, 128U);
    const MatrixRunDesc output_desc = make_run_desc(output_head_row_elems, output_head_row_count, output_head_row_count);

    std::vector<float> prompt_rows;
    std::vector<float> hidden;
    std::vector<float> normed_hidden;
    std::vector<float> logits;
    std::vector<float> draft_out;
    std::vector<float> head_out;
    VectorStats target_stats{};
    VectorStats draft_stats{};
    VectorStats head_stats{};
    ArgmaxResult target_argmax{};
    ArgmaxResult draft_argmax{};
    ArgmaxResult head_argmax{};

    if (prompt_ids && prompt_id_count > 0) {
        // 1. Gather token embeddings for all prompt tokens
        if (!run_q4k_gather(target_tensor,
                            target_tensor_bytes,
                            target_row_elems,
                            prompt_ids,
                            prompt_id_count,
                            prompt_rows,
                            error,
                            error_cap)) {
            return false;
        }
        // 2. Mean-pool embeddings to get a single hidden state vector
        if (!mean_rows(prompt_rows, prompt_id_count, target_row_elems, hidden)) {
            set_error(error, error_cap, "failed to average prompt embeddings");
            return false;
        }
        // 3. RMS-normalize the hidden state
        if (!run_rms_norm(hidden, normed_hidden, error, error_cap)) {
            return false;
        }
        // 4. Project through the LM head (output.weight) to get vocabulary logits
        if (!run_q6k_matvec(output_head_tensor,
                            output_head_tensor_bytes,
                            output_desc,
                            normed_hidden,
                            head_out,
                            error,
                            error_cap) ||
            !analyze_vector(head_out, head_stats, head_argmax, error, error_cap)) {
            return false;
        }
        // 5. Generate token IDs from the LM head logits
        if (generated_ids && generated_cap > 0) {
            const size_t produced = collect_top_k(head_out, generated_cap, generated_ids, generated_cap);
            if (generated_count) {
                *generated_count = produced;
            }
        }
        // 6. Diagnostic: embedding-space similarity scores via target embedding table
        if (!run_q4k_matvec(target_tensor,
                            target_tensor_bytes,
                            make_run_desc(target_row_elems, target_row_count, target_row_count),
                            normed_hidden,
                            logits,
                            error,
                            error_cap) ||
            !analyze_vector(logits, target_stats, target_argmax, error, error_cap)) {
            return false;
        }
    }

    // 7. Draft diagnostic: run a Q8_0 matvec on the hidden state (or synthetic if no context)
    {
        const bool use_hidden = !normed_hidden.empty() && normed_hidden.size() == draft_row_elems;
        const std::vector<float>& draft_input = use_hidden ? normed_hidden
            : (prompt_ids && prompt_id_count > 0
               ? make_prompt_token_vector(prompt_ids, prompt_id_count, draft_row_elems)
               : make_prompt_vector(prompt, draft_row_elems));
        std::vector<float> draft_normed_tmp;
        const std::vector<float>& draft_normed = use_hidden ? normed_hidden : draft_normed_tmp;
        if (!use_hidden) {
            if (!run_rms_norm(draft_input, draft_normed_tmp, error, error_cap)) {
                return false;
            }
        }
        if (!run_q80_matvec(draft_tensor,
                            draft_tensor_bytes,
                            draft_desc,
                            draft_normed,
                            draft_out,
                            error,
                            error_cap) ||
            !analyze_vector(draft_out, draft_stats, draft_argmax, error, error_cap)) {
            return false;
        }
    }

    const float ty0 = logits.size() > 0 ? logits[0] : 0.0f;
    const float ty1 = logits.size() > 1 ? logits[1] : 0.0f;
    const float dy0 = draft_out.size() > 0 ? draft_out[0] : 0.0f;
    const float dy1 = draft_out.size() > 1 ? draft_out[1] : 0.0f;
    const float hy0 = head_out.size() > 0 ? head_out[0] : 0.0f;
    const float hy1 = head_out.size() > 1 ? head_out[1] : 0.0f;

    std::snprintf(output,
                  output_cap,
                  "q4_probe=%0.4f head_argmax=%d/%0.3f head_mean=%0.4f targ_argmax=%d/%0.3f draft_argmax=%d/%0.3f"
                  " tgt_rows=%u dft_rows=%u head_rows=%u tok=%zu gen=%zu hy0=%0.3f hy1=%0.3f",
                  q4_probe,
                  head_argmax.index,
                  head_argmax.value,
                  head_stats.mean_abs,
                  target_argmax.index,
                  target_argmax.value,
                  draft_argmax.index,
                  draft_argmax.value,
                  static_cast<unsigned>(target_desc.rows_to_run),
                  static_cast<unsigned>(draft_desc.rows_to_run),
                  static_cast<unsigned>(output_desc.rows_to_run),
                  prompt_id_count,
                  generated_count ? *generated_count : 0U,
                  hy0,
                  hy1);
    return true;
}
