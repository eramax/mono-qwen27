Mono27B Engine: RTX 3090 Kernel Fusion & Optimization GuideFor single-token generation (batch size 1), the RTX 3090's INT32/FP16 compute power far exceeds what we can feed it. Our goal is to achieve maximum memory bus utilization (~936 GB/s). Every time we write an intermediate tensor to global memory and read it back in the next kernel, we waste precious time.Here are 5 high-impact optimizations targeting your codebase.1. Vectorize Memory Accesses (128-bit Loads/Stores)Currently, element-wise operations like k_elem_add, k_elem_mul, and k_elem_silu use scalar floats (32-bit). The RTX 3090 memory controller is most efficient when moving 128-bit chunks (float4).Improvement: Recast your element-wise kernels to process float4. This yields an almost instant 2x–3x speedup on these operations.// Fused and Vectorized Element Add
__global__ static void k_elem_add_float4(float * a, const float * b, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < n) {
        float4* va_ptr = reinterpret_cast<float4*>(&a[i]);
        const float4* vb_ptr = reinterpret_cast<const float4*>(&b[i]);
        
        float4 va = *va_ptr;
        float4 vb = *vb_ptr;
        
        va.x += vb.x; va.y += vb.y; va.z += vb.z; va.w += vb.w;
        *va_ptr = va;
    }
}

Note: Make sure your tensor dimensions (like MONO27B_TARGET_HIDDEN = 5120) are divisible by 4, which they are.2. Epilogue Fusion (MatVec + Residual Add)Current Flow: 1. MV(L.wo, qb, h2) — MatVec writes out to h2 2. k_elem_add(h2, h) — Reads h2 and h, adds them, writes back to h2Improvement: Pass the residual pointer directly into the DP4A matrix-vector multiplication kernels. This saves an entire global memory round-trip for the h2 tensor.Modify your MatVec kernels (like k_q4k_mv_q8_dp4a) to include a const float * residual parameter:__global__ static void k_q4k_mv_q8_dp4a(
    const BlockQ4K * __restrict__ W, 
    const BlockQ8_1 * __restrict__ q8, 
    float * __restrict__ y, 
    const float * __restrict__ residual, // <-- NEW
    int rb, int rc) 
{
    // ... [existing reduction logic] ...
    
    if (threadIdx.y == 0 && threadIdx.x < NWARPS) {
        sum = smem[threadIdx.x];
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            // FUSED EPILOGUE
            y[row] = sum + (residual ? residual[row] : 0.0f);
        }
    }
}

3. Fused M-RoPE and KV Cache UpdateCurrent Flow:k_mrope(kb, ...) — Reads kb, applies rotations, writes to kb.k_write_kv_cache(kb, ...) — Reads kb, writes to kv_cache_k and kv_cache_v.Improvement: The M-RoPE kernel should directly write its final result into the KV Cache. kb becomes completely ephemeral (or only used as temporary workspace).__global__ static void k_mrope_and_cache(
    const float * __restrict__ buf_in, 
    float * __restrict__ kv_k_out,
    float * __restrict__ kv_v_out,
    int n_heads, int head_dim, 
    int sec0, int sec1, int sec2, int sec3, 
    int n_rot_dims, int kv_pos)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    
    // ... [calculate RoPE for buf_in exactly as before] ...
    // float v0 = hd[d0], v1 = hd[d1];
    // float new_d0 = v0 * c - v1 * s;
    // float new_d1 = v0 * s + v1 * c;
    
    // INSTEAD of writing back to hd, calculate cache offsets directly:
    size_t off_k = ((size_t)kv_pos * n_heads + h) * head_dim + d0;
    size_t off_v = ((size_t)kv_pos * n_heads + h) * head_dim + d0; // (Offset for V)

    // Write straight to global KV cache
    kv_k_out[off_k] = new_d0;
    // ... handling V caching simultaneously
}

4. Fuse Sigmoid + Multiplication + Q8_1 QuantizationBefore the L.wo (Attention Output) matvec, you compute the gate via sigmoid and multiply it by the output, then quantize it to DP4A.Current: k_elem_sigmoid_mul(qb) -> k_quant_q8_1(qb, g_q8_scratch)Improvement: Create a fused quantization kernel that computes the sigmoid activation on the fly in registers while doing the block max/sum reduction for DP4A.__launch_bounds__(32, 1)
__global__ static void k_quant_q8_1_fused_gate(
    const float * __restrict__ x, 
    const float * __restrict__ gate,
    BlockQ8_1 * __restrict__ y, 
    int n) 
{
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= n) return;

    // FUSED Sigmoid + Mul
    float sig = 1.0f / (1.0f + __expf(-gate[i0]));
    const float xi = x[i0] * sig;
    
    // DP4A Quantization logic
    const int ib  = i0 >> 5;
    const int iqs = i0 & 31;
    float amax = fabsf(xi);
    float sum  = xi;

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum  += __shfl_xor_sync(0xffffffff, sum, o);
    }

    const float d = amax / 127.0f;
    y[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

5. Parallelizing QKV DispatchCurrently, L.wq, L.wk, and L.wv are dispatched sequentially. Even though they are encapsulated in a CUDA graph, the SMs on the 3090 (82 of them) might not be fully saturated by the smaller K and V projections (since MQA/GQA reduces K and V dimension).Improvement: You have state->stream1 allocated. Utilize it to launch the projections in parallel, which overlap their memory loads.// Ensure h2 is quantized ONCE into g_q8_scratch
int n_q8 = l_n_q8_for(L.wq.ggml_type, L.wq.row_blocks);
l_quant_q8_n(h2, n_q8);

// Launch WQ on stream 0
l_mv_q8_on(L.wq.ptr, L.wq.ggml_type, L.wq.row_blocks, L.wq.row_count, qb, 0);

// Launch WK and WV on stream 1
l_mv_q8_on(L.wk.ptr, L.wk.ggml_type, L.wk.row_blocks, L.wk.row_count, kb, (cudaStream_t)st->stream1);
l_mv_q8_on(L.wv.ptr, L.wv.ggml_type, L.wv.row_blocks, L.wv.row_count, kb + MONO27B_TARGET_KV_DIM, (cudaStream_t)st->stream1);

// Synchronize streams before RoPE
cudaEventRecord((cudaEvent_t)st->sync_event, (cudaStream_t)st->stream1);
cudaStreamWaitEvent(0, (cudaEvent_t)st->sync_event, 0);

Note: This strategy perfectly interfaces with your existing CUDA graphs. Graph captures will absorb these parallel streams as concurrent nodes, natively building an optimized execution topology.




<!-- ... existing code ... -->
    const float d = amax / 127.0f;
    y[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}

__launch_bounds__(32, 1)
__global__ static void k_quant_q8_1_fused_gate(
    const float * __restrict__ x, 
    const float * __restrict__ gate,
    BlockQ8_1 * __restrict__ y, 
    int n) 
{
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= n) return;

    // FUSED Sigmoid + Mul
    float sig = 1.0f / (1.0f + __expf(-gate[i0]));
    const float xi = x[i0] * sig;
    
    // DP4A Quantization logic
    const int ib  = i0 >> 5;
    const int iqs = i0 & 31;
    float amax = fabsf(xi);
    float sum  = xi;

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum  += __shfl_xor_sync(0xffffffff, sum, o);
    }

    const float d = amax / 127.0f;
    y[ib].qs[iqs] = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

// ─── dp4a helper (from ggml common.cuh) ─────────────────────────────────────
<!-- ... existing code ... -->
// Key insight: for single-token inference (1 row), parallel warp reductions are faster
// than llama.cpp's sequential accumulation in warp 0
// Grid: rc blocks (one per output row)
// Block: (32, 4) = 128 threads total (4 warps)
__launch_bounds__(128, 1)
__global__ static void k_q4k_mv_q8_dp4a(const BlockQ4K * __restrict__ W, const BlockQ8_1 * __restrict__ q8, float * __restrict__ y, const float * __restrict__ residual, int rb, int rc) {
    constexpr int VDR = 2;       // Vector Dot Ratio
    constexpr int QI = 32;       // Q4_K quant dimension
<!-- ... existing code ... -->
    if (threadIdx.y == 0 && threadIdx.x < NWARPS) {
        sum = smem[threadIdx.x];
        #pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = sum + (residual ? residual[row] : 0.0f);
    }
}

// Fused gate+up Q4_K matvec with SwiGLU (silu(gate) * up). Replaces 3 launches
<!-- ... existing code ... -->
// ─── Elementwise kernels ─────────────────────────────────────────────────────

__global__ static void k_elem_add(float * a, const float * b, int n) {
    int limit = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < limit; i += gridDim.x * blockDim.x) {
        float4* va_ptr = reinterpret_cast<float4*>(&a[i * 4]);
        const float4* vb_ptr = reinterpret_cast<const float4*>(&b[i * 4]);
        float4 va = *va_ptr;
        float4 vb = *vb_ptr;
        va.x += vb.x; va.y += vb.y; va.z += vb.z; va.w += vb.w;
        *va_ptr = va;
    }
    int tail_start = limit * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        a[i] += b[i];
    }
}

// Fused copy + residual: out[i] = src[i] + res[i]
__global__ static void k_elem_copy_add(const float * src, const float * res, float * out, int n) {
    int limit = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < limit; i += gridDim.x * blockDim.x) {
        const float4* vsrc = reinterpret_cast<const float4*>(&src[i * 4]);
        const float4* vres = reinterpret_cast<const float4*>(&res[i * 4]);
        float4* vout = reinterpret_cast<float4*>(&out[i * 4]);
        float4 s = *vsrc;
        float4 r = *vres;
        s.x += r.x; s.y += r.y; s.z += r.z; s.w += r.w;
        *vout = s;
    }
    int tail_start = limit * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = src[i] + res[i];
    }
}

__global__ static void k_elem_mul(float * a, const float * b, int n) {
    int limit = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < limit; i += gridDim.x * blockDim.x) {
        float4* va_ptr = reinterpret_cast<float4*>(&a[i * 4]);
        const float4* vb_ptr = reinterpret_cast<const float4*>(&b[i * 4]);
        float4 va = *va_ptr;
        float4 vb = *vb_ptr;
        va.x *= vb.x; va.y *= vb.y; va.z *= vb.z; va.w *= vb.w;
        *va_ptr = va;
    }
    int tail_start = limit * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        a[i] *= b[i];
    }
}

__global__ static void k_elem_copy(float * d, const float * s, int n) {
    int limit = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < limit; i += gridDim.x * blockDim.x) {
        float4* vd_ptr = reinterpret_cast<float4*>(&d[i * 4]);
        const float4* vs_ptr = reinterpret_cast<const float4*>(&s[i * 4]);
        *vd_ptr = *vs_ptr;
    }
    int tail_start = limit * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        d[i] = s[i];
    }
}

__global__ static void k_elem_silu(float * x, int n) {
    int limit = n / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < limit; i += gridDim.x * blockDim.x) {
        float4* vx_ptr = reinterpret_cast<float4*>(&x[i * 4]);
        float4 vx = *vx_ptr;
        vx.x = vx.x / (1.0f + expf(-vx.x));
        vx.y = vx.y / (1.0f + expf(-vx.y));
        vx.z = vx.z / (1.0f + expf(-vx.z));
        vx.w = vx.w / (1.0f + expf(-vx.w));
        *vx_ptr = vx;
    }
    int tail_start = limit * 4;
    for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ static void k_elem_softplus(const float * x, const float * bias, float * out, int n) {
<!-- ... existing code ... -->
    // Phase 3: weighted sum of V values
    float out_val = 0.0f;
    for (int p = 0; p < kv_len; ++p) {
        float sv = scores[p] * inv_sum;
        out_val += sv * V[(size_t)p * pos_stride + (size_t)kvh * hd + t];
    }
    or_[t] = out_val;
}

__global__ static void k_mrope_and_cache(
    const float * __restrict__ kb_in, 
    float * __restrict__ kv_k_out,
    float * __restrict__ kv_v_out,
    int n_heads, int head_dim, 
    int sec0, int sec1, int sec2, int sec3, 
    int n_rot_dims)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    
    const float * hd = kb_in + (size_t)h * head_dim;
    const float * vd = kb_in + (size_t)n_heads * head_dim + (size_t)h * head_dim;

    const int sec01 = sec0 + sec1;
    const int sec012 = sec01 + sec2;
    const int n_rot_pairs = n_rot_dims / 2;
    int pos = g_kv_pos;

    for (int p = threadIdx.x; p < head_dim; p += blockDim.x) {
        float k_val = hd[p];
        float v_val = vd[p];

        if (p < n_rot_dims) {
            int pair_idx = p < n_rot_pairs ? p : p - n_rot_pairs;
            int is_upper = p >= n_rot_pairs;
            
            int pos_stream = 0;
            if (pair_idx < sec0) pos_stream = pos;
            else if (pair_idx < sec01) pos_stream = pos;
            else if (pair_idx < sec012) pos_stream = pos;

            float theta = powf(MONO27B_TARGET_ROPE_THETA, -2.0f * (float)pair_idx / (float)n_rot_dims);
            float c = cosf(theta * (float)pos_stream), s = sinf(theta * (float)pos_stream);
            
            float v0 = hd[pair_idx];
            float v1 = hd[pair_idx + n_rot_pairs];
            
            if (!is_upper) {
                k_val = v0 * c - v1 * s;
            } else {
                k_val = v0 * s + v1 * c;
            }
        }

        size_t off = ((size_t)pos * n_heads + h) * head_dim + p;
        kv_k_out[off] = k_val;
        kv_v_out[off] = v_val;
    }
}

// ─── SSM conv1d (1 thread per channel) ───────────────────────────────────────
<!-- ... existing code ... -->
}

// Matvec using ALREADY QUANTIZED Q8_1 data in g_q8_scratch
// stream = 0 for default stream, or a custom CUDA stream for concurrency
static void l_mv_q8_on(void * W, uint32_t ggml_type, int rb, int rc, float * y, const float * residual, cudaStream_t stream) {
    if (rc == 0 || !W) return;
    if (!g_q8_scratch) return;
    int n_q8 = l_n_q8_for(ggml_type, rb);
    if (n_q8 > g_kernel_cfg.q8_scratch_max_blocks) return;
    switch (ggml_type) {
        case MONO27B_GGML_TYPE_Q4_K:
            k_q4k_mv_q8_dp4a<<<rc, dim3(32, g_kernel_cfg.q4k_q8_warp_count),
                32 * (size_t)g_kernel_cfg.q4k_q8_smem_per_warp * sizeof(float), stream>>>(
                (const BlockQ4K *)W, g_q8_scratch, y, residual, rb, rc);
            return;
        case MONO27B_GGML_TYPE_Q5_K:
            k_q5k_mv_q8<<<rc, dim3(32, 4), 0, stream>>>((const BlockQ5K *)W, g_q8_scratch, y, rb, rc);
            if (residual) k_elem_add<<<(rc + 255)/256, 256, 0, stream>>>(y, residual, rc);
            return;
        case MONO27B_GGML_TYPE_Q6_K:
            k_q6k_mv_q8_dp4a<<<rc, 128, 0, stream>>>((const BlockQ6K *)W, g_q8_scratch, y, rb, rc);
            if (residual) k_elem_add<<<(rc + 255)/256, 256, 0, stream>>>(y, residual, rc);
            return;
        case MONO27B_GGML_TYPE_Q8_0:
            k_q80_mv_q8_dp4a<<<rc, dim3(32, 4), 0, stream>>>((const BlockQ8 *)W, g_q8_scratch, y, rb, rc);
            if (residual) k_elem_add<<<(rc + 255)/256, 256, 0, stream>>>(y, residual, rc);
            return;
        case 23:
            k_iq4xs_mv_q8_dp4a<<<rc, g_kernel_cfg.q4k_q8_threads, 0, stream>>>((const BlockIQ4XS *)W, g_q8_scratch, y, rb, rc);
            if (residual) k_elem_add<<<(rc + 255)/256, 256, 0, stream>>>(y, residual, rc);
            return;
        default: break;
    }
}

static void l_mv_q8(void * W, uint32_t ggml_type, int rb, int rc, float * y) {
    l_mv_q8_on(W, ggml_type, rb, rc, y, nullptr, 0);
}

// Launch two matvecs concurrently on the same Q8_1 input.
<!-- ... existing code ... -->
    int n_q8 = std::max(n_q8_1, n_q8_2);  // both pair members must operate on the same input
    if (g_q8_scratch && n_q8 <= g_kernel_cfg.q8_scratch_max_blocks && n_q8_1 == n_q8_2) {
        l_quant_q8_n(x, n_q8);
        l_mv_q8_on(w1, t1, rb1, rc1, y1, nullptr, 0);
        l_mv_q8_on(w2, t2, rb2, rc2, y2, nullptr, 0);
        return;
    }
    // Fallback: sequential on stream 0
<!-- ... existing code ... -->
        if (is_a) {
            const auto & L = we->layers[il].attn;
            int max_ctx = st->max_ctx > 0 ? st->max_ctx : 8192;

            l_rms(h2, h, WV(L.attn_norm), MONO27B_TARGET_HIDDEN); TRACE("rms");
            
            int n_q8_q = l_n_q8_for(L.wq.ggml_type, L.wq.row_blocks);
            l_quant_q8_n(h2, n_q8_q);

            l_mv_q8_on(L.wq.ptr, L.wq.ggml_type, L.wq.row_blocks, L.wq.row_count, qb, nullptr, 0);
            l_mv_q8_on(L.wk.ptr, L.wk.ggml_type, L.wk.row_blocks, L.wk.row_count, kb, nullptr, (cudaStream_t)st->stream1);
            l_mv_q8_on(L.wv.ptr, L.wv.ggml_type, L.wv.row_blocks, L.wv.row_count, kb + MONO27B_TARGET_KV_DIM, nullptr, (cudaStream_t)st->stream1);

            cudaEventRecord((cudaEvent_t)st->sync_event, (cudaStream_t)st->stream1);
            cudaStreamWaitEvent(0, (cudaEvent_t)st->sync_event, 0);
            TRACE("qkv");

            // Qwen3Next: Q projection outputs interleaved Q+gate per head:
            // [Q0(256), G0(256), Q1(256), G1(256), ...]
            // Deinterleave to contiguous Q (6144) and gate (6144).
            // Use fb as a temporary source buffer to avoid in-place race
            // (head N overwrites src data that head 0..N-1 still need to read).
            {
                cudaMemcpyAsync(fb, qb, MONO27B_TARGET_Q_DIM * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
                int n_h = MONO27B_TARGET_N_HEAD;
                int hd = MONO27B_TARGET_HEAD_DIM;
                k_deinterleave_qg<<<n_h, hd>>>(fb, qb, qb + MONO27B_TARGET_Q_DIM, n_h, hd);
            }
            if (L.q_norm.ptr)
<!-- ... existing code ... -->
            if (L.k_norm.ptr)
                k_rms_norm_mulw_batched<<<MONO27B_TARGET_N_KV_HEAD, g_kernel_cfg.rms_norm_threads>>>(
                    kb, WV(L.k_norm), kb, MONO27B_TARGET_HEAD_DIM, MONO27B_TARGET_N_KV_HEAD, g_kernel_cfg.rms_eps);
            TRACE("qkn");

            // M-RoPE: text tokens use the same token position for the first
            // three position streams and 0 for the 4th stream.
            k_mrope<<<MONO27B_TARGET_N_HEAD, 64>>>(
                qb, MONO27B_TARGET_N_HEAD, MONO27B_TARGET_HEAD_DIM,
                11, 11, 10, 0,
                MONO27B_N_ROT_DIMS); TRACE("mq");
                
            k_mrope_and_cache<<<MONO27B_TARGET_N_KV_HEAD, MONO27B_TARGET_HEAD_DIM>>>(
                kb, st->kv_cache_k[fa_i], st->kv_cache_v[fa_i],
                MONO27B_TARGET_N_KV_HEAD, MONO27B_TARGET_HEAD_DIM,
                11, 11, 10, 0,
                MONO27B_N_ROT_DIMS);
            TRACE("mk+kvc");

            // Attention: parallel cooperative kernel
            {
                int smem_kvl = max_ctx > 0 ? max_ctx : 8192;
<!-- ... existing code ... -->
                    MONO27B_TARGET_HEAD_DIM, max_ctx, 1.0f / sqrtf(MONO27B_TARGET_HEAD_DIM));
            }
            
            // Gate: sigmoid(gate_q) * attn_output + DP4A Quantization
            int n_q8_wo = l_n_q8_for(L.wo.ggml_type, L.wo.row_blocks);
            k_quant_q8_1_fused_gate<<<n_q8_wo, 32>>>(qb, qb + MONO27B_TARGET_Q_DIM, g_q8_scratch, MONO27B_TARGET_Q_DIM);
            TRACE("gt+quant");

            l_mv_q8_on(L.wo.ptr, L.wo.ggml_type, L.wo.row_blocks, L.wo.row_count, h2, h, 0); // h is residual
            TRACE("wo+res1");

            l_rms(h, h2, WV(L.post_norm), MONO27B_TARGET_HIDDEN); TRACE("porm");
<!-- ... existing code ... -->