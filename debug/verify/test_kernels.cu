// Standalone CUDA kernel verification tests for mono27b
// These tests compare our custom kernels against reference formulas copied
// directly from llama.cpp / ggml to ensure parity.
//
// Build: nvcc -o test_kernels test_kernels.cu -arch=sm_86 -I../../include
// Run:  ./test_kernels

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

constexpr float EPS = 1e-6f;

// ─── Helpers ────────────────────────────────────────────────────────────────
static float cpu_silu(float x) { return x / (1.0f + expf(-x)); }

static float cpu_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float cpu_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// ─── Reference L2 norm (matches ggml-cuda/norm.cu l2_norm_f32) ─────────────
static void l2_norm_ref(const float * x, float * dst, int ncols, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < ncols; ++i) sum_sq += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf(fmaxf((float)sum_sq, eps * eps));
    for (int i = 0; i < ncols; ++i) dst[i] = scale * x[i];
}

// ─── Reference RMS norm (matches ggml-cuda/norm.cu rms_norm_f32) ───────────
static void rms_norm_ref(const float * x, float * dst, int ncols, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < ncols; ++i) sum_sq += (double)x[i] * (double)x[i];
    float mean = (float)sum_sq / (float)ncols;
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < ncols; ++i) dst[i] = scale * x[i];
}

// ─── Reference conv1d + SiLU (matches ggml-cuda/ssm-conv.cu) ───────────────
static void ssm_conv1d_ref(const float * inp, const float * w, float * cs,
                           float * out, int cc, int ck) {
    int sl = ck - 1;
    for (int ch = 0; ch < cc; ++ch) {
        float s = 0.0f;
        for (int k = 0; k < sl; ++k) s += cs[ch * sl + k] * w[ch * ck + k];
        s += inp[ch] * w[ch * ck + sl];
        for (int k = 0; k < sl - 1; ++k) cs[ch * sl + k] = cs[ch * sl + k + 1];
        cs[ch * sl + (sl - 1)] = inp[ch];
        out[ch] = cpu_silu(s);
    }
}

// ─── Reference DeltaNet (matches ggml-cuda/gated_delta_net.cu non-KDA) ─────
static void deltanet_ref(const float * q, const float * k, const float * v,
                         const float * g, const float * beta,
                         float * state, float * out,
                         int ng, int dr, int hv, int hk) {
    float scale = 1.0f / sqrtf((float)hv);
    for (int r_idx = 0; r_idx < dr; ++r_idx) {
        int g_idx = r_idx / (dr / ng);
        float gv = expf(g[r_idx]);
        float bv = beta[r_idx];
        for (int col = 0; col < hv; ++col) {
            float * S = state + r_idx * hv * hk + col * hk;
            float kv = 0.0f;
            for (int i = 0; i < hk; ++i) kv += S[i] * k[g_idx * hk + i];
            float delta = (v[r_idx * hv + col] - gv * kv) * bv;
            float attn = 0.0f;
            for (int i = 0; i < hk; ++i) {
                S[i] = gv * S[i] + k[g_idx * hk + i] * delta;
                attn += S[i] * q[g_idx * hk + i];
            }
            out[r_idx * hv + col] = attn * scale;
        }
    }
}

// ─── Our kernels (copied from mono27b_executor.cu) ─────────────────────────
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
    const float scale = rsqrtf(fmaxf((float)sum_sq, EPS * EPS));
    for (int i = threadIdx.x; i < gs; i += 128) x[i] *= scale;
}

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

__global__ static void k_elem_silu(float * x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

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

__global__ static void k_deltanet(
    const float * q, const float * k, const float * v,
    const float * g, const float * beta,
    float * state, float * out,
    int ng, int dr, int hv, int hk)
{
    int r_idx = blockIdx.x;
    int col   = blockIdx.y;
    if (r_idx >= dr || col >= hv) return;
    int g_idx = r_idx / (dr / ng);
    float gv = expf(g[r_idx]);
    float bv = beta[r_idx];
    const float * qg = q + (size_t)g_idx * hk;
    const float * kg = k + (size_t)g_idx * hk;
    const float * vr = v + (size_t)r_idx * hv + col;
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
    out[(size_t)r_idx * hv + col] = attn * (1.0f / sqrtf((float)hv));
}

// ─── Test harness ───────────────────────────────────────────────────────────
static bool test_l2_norm() {
    const int n_groups = 16;
    const int group_size = 128;
    const int n = n_groups * group_size;
    std::vector<float> h_x(n), h_ref(n), h_gpu(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) h_x[i] = dist(rng);
    for (int g = 0; g < n_groups; ++g)
        l2_norm_ref(h_x.data() + g * group_size, h_ref.data() + g * group_size, group_size, EPS);
    float * d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    k_l2_norm_g<<<n_groups, 128>>>(d_x, group_size, n_groups);
    cudaMemcpy(h_gpu.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) max_diff = std::max(max_diff, (double)std::fabs(h_ref[i] - h_gpu[i]));
    printf("L2 norm test:     max_diff = %.9g %s\n", max_diff, max_diff < 1e-5 ? "PASS" : "FAIL");
    return max_diff < 1e-5;
}

static bool test_rms_norm() {
    const int n = 5120;
    std::vector<float> h_x(n), h_w(n), h_ref(n), h_gpu(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) { h_x[i] = dist(rng); h_w[i] = 1.0f + dist(rng) * 0.1f; }
    rms_norm_ref(h_x.data(), h_ref.data(), n, EPS);
    for (int i = 0; i < n; ++i) h_ref[i] *= h_w[i];
    float *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    k_rms_norm_mulw<256><<<1, 256>>>(d_x, d_w, d_y, n, EPS);
    cudaMemcpy(h_gpu.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) max_diff = std::max(max_diff, (double)std::fabs(h_ref[i] - h_gpu[i]));
    printf("RMS norm test:    max_diff = %.9g %s\n", max_diff, max_diff < 1e-5 ? "PASS" : "FAIL");
    return max_diff < 1e-5;
}

static bool test_conv1d_silu() {
    const int cc = 10240;
    const int ck = 4;
    const int sl = ck - 1;
    std::vector<float> h_inp(cc), h_w(cc * ck), h_cs(cc * sl), h_ref(cc), h_gpu(cc);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < cc; ++i) h_inp[i] = dist(rng);
    for (int i = 0; i < cc * ck; ++i) h_w[i] = dist(rng);
    for (int i = 0; i < cc * sl; ++i) h_cs[i] = dist(rng);
    std::vector<float> cs_copy = h_cs;
    ssm_conv1d_ref(h_inp.data(), h_w.data(), cs_copy.data(), h_ref.data(), cc, ck);

    float *d_inp, *d_w, *d_cs, *d_out;
    cudaMalloc(&d_inp, cc * sizeof(float));
    cudaMalloc(&d_w, cc * ck * sizeof(float));
    cudaMalloc(&d_cs, cc * sl * sizeof(float));
    cudaMalloc(&d_out, cc * sizeof(float));
    cudaMemcpy(d_inp, h_inp.data(), cc * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), cc * ck * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cs, h_cs.data(), cc * sl * sizeof(float), cudaMemcpyHostToDevice);
    k_ssm_conv1d_u<<<(cc + 255) / 256, 256>>>(d_inp, d_w, d_cs, d_out, cc, ck);
    k_elem_silu<<<(cc + 255) / 256, 256>>>(d_out, cc);
    cudaMemcpy(h_gpu.data(), d_out, cc * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_inp); cudaFree(d_w); cudaFree(d_cs); cudaFree(d_out);

    double max_diff = 0.0;
    for (int i = 0; i < cc; ++i) max_diff = std::max(max_diff, (double)std::fabs(h_ref[i] - h_gpu[i]));
    printf("Conv1D+SiLU test: max_diff = %.9g %s\n", max_diff, max_diff < 1e-4 ? "PASS" : "FAIL");
    return max_diff < 1e-4;
}

static bool test_deltanet() {
    const int ng = 16, dr = 48, hv = 128, hk = 128;
    const int n_qk = ng * hk;
    const int n_v = dr * hv;
    const int state_sz = dr * hv * hk;
    std::vector<float> h_q(n_qk), h_k(n_qk), h_v(n_v), h_g(dr), h_b(dr);
    std::vector<float> h_state(state_sz), h_ref(n_v), h_gpu(n_v);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto & x : h_q) x = dist(rng);
    for (auto & x : h_k) x = dist(rng);
    for (auto & x : h_v) x = dist(rng);
    for (auto & x : h_g) x = dist(rng) * 0.1f;
    for (auto & x : h_b) x = dist(rng);
    for (auto & x : h_state) x = 0.0f;

    std::vector<float> state_ref = h_state;
    deltanet_ref(h_q.data(), h_k.data(), h_v.data(), h_g.data(), h_b.data(),
                 state_ref.data(), h_ref.data(), ng, dr, hv, hk);

    float *d_q, *d_k, *d_v, *d_g, *d_b, *d_state, *d_out;
    cudaMalloc(&d_q, n_qk * sizeof(float));
    cudaMalloc(&d_k, n_qk * sizeof(float));
    cudaMalloc(&d_v, n_v * sizeof(float));
    cudaMalloc(&d_g, dr * sizeof(float));
    cudaMalloc(&d_b, dr * sizeof(float));
    cudaMalloc(&d_state, state_sz * sizeof(float));
    cudaMalloc(&d_out, n_v * sizeof(float));
    cudaMemcpy(d_q, h_q.data(), n_qk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), n_qk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), n_v * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g.data(), dr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), dr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state, h_state.data(), state_sz * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dg(dr, hv, 1);
    k_deltanet<<<dg, 1>>>(d_q, d_k, d_v, d_g, d_b, d_state, d_out, ng, dr, hv, hk);
    cudaMemcpy(h_gpu.data(), d_out, n_v * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_b); cudaFree(d_state); cudaFree(d_out);

    double max_diff = 0.0;
    for (int i = 0; i < n_v; ++i) max_diff = std::max(max_diff, (double)std::fabs(h_ref[i] - h_gpu[i]));
    printf("DeltaNet test:    max_diff = %.9g %s\n", max_diff, max_diff < 1e-4 ? "PASS" : "FAIL");
    return max_diff < 1e-4;
}

int main() {
    bool ok = true;
    ok &= test_l2_norm();
    ok &= test_rms_norm();
    ok &= test_conv1d_silu();
    ok &= test_deltanet();
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
