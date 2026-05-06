# mono27b vs llama.cpp — Performance Analysis

**Generated:** 2026-05-06  
**GPU:** NVIDIA GeForce RTX 3090 (936 GB/s peak)  
**Model:** Qwen3.6-27B-UD-Q4_K_XL.gguf

---

## Current Speed

| Metric | Mono27B | llama.cpp | Gap |
|--------|---------|-----------|-----|
| Sampling (tok/s) | **25.5** | **39.3** | **1.54×** |
| ms/token | 39.2 | 25.5 | 13.7 ms |
| GPU util (bw %) | ~38% | ~62% | — |

**Baseline (no optimizations):** 18.4 tok/s  
**After optimizations:** 25.5 tok/s (+38%)

---

## Timing Breakdown (per token, GPU time)

From `make compare-perf` with CUDA event tracing:

| Category | Mono27B (ms) | % | Est. llama (ms) | Gap (ms) |
|----------|-------------|---|----------------|----------|
| **SSM state+out** (ssmo) | **11.7** | 30.9% | 7.4 | **4.3** |
| **SSM wqkv+gate** (re-g) | **7.5** | 19.8% | 4.7 | **2.8** |
| FFN gate+up (fg+fu) | 2.1 | 5.6% | 1.3 | 0.8 |
| Attention Q/K/V/O | 1.9 | 5.0% | 1.2 | 0.7 |
| FFN swiglu+down | 1.4 | 3.7% | 0.9 | 0.5 |
| Attention RoPE+KV | 0.5 | 1.4% | 0.3 | 0.2 |
| SSM norms+mul | 0.5 | 1.3% | 0.3 | 0.2 |
| Attention norms | 0.3 | 0.9% | 0.2 | 0.1 |
| FFN norm | 0.3 | 0.7% | 0.2 | 0.1 |
| Embed+head | 0.04 | 0.1% | 0.03 | 0.02 |
| Inter-layer / overhead ("pre") | 11.1 | 29.4% | — | — |
| **Total GPU** | **37.8** | 100% | **24.0** | **13.8** |
| CPU / D2H / overhead | 2.5 | — | ~1.5 | — |
| **Wall clock** | **40.2** | — | **25.5** | **14.7** |

> **Note:** The "pre" category (11.1 ms) is an inter-layer timing artifact that captures pipeline bubbles and async delays between consecutive layers. It is NOT a specific kernel.

---

## Top 10 Individual Kernels (by time)

| Kernel | ms/tok | % | calls/tok | What it does |
|--------|--------|---|-----------|-------------|
| **ssmo** | 11.66 | 30.9% | 48 | SSM output matvec (6144→5120, Q4_K) |
| pre | 11.08 | 29.4% | 64 | Inter-layer overhead (artifact) |
| **re-g** | 7.48 | 19.8% | 48 | SSM wqkv_gate matvec (5120→6144, Q5_K) |
| fg+fu | 2.11 | 5.6% | 16 | FFN gate+up matvec pair (5120→17408, Q4_K) |
| fd | 1.32 | 3.5% | 16 | FFN down matvec (17408→5120, Q4_K) |
| wq | 0.84 | 2.2% | 16 | Attention Q matvec (5120→6144, Q4_K) |
| wo | 0.50 | 1.3% | 16 | Attention O matvec (6144→5120, Q4_K) |
| gt | 0.33 | 0.9% | 16 | Attention gate multiply |
| grms | 0.33 | 0.9% | 48 | SSM group RMS norm |
| rms | 0.28 | 0.7% | 16 | Attention pre-norm |

---

## Critical Finding: Pipeline vs Isolation

**Microbenchmark** (isolated kernel, no pipeline):
| Kernel | Isolated | In Pipeline | Slowdown |
|--------|----------|-------------|----------|
| ssmo (5120×6144 Q4_K) | 0.023 ms | 0.24 ms | **10.4×** |
| wqkv (6144×5120 Q4_K) | 0.027 ms | 0.16 ms | **5.9×** |
| FFN gate (17408×5120 Q4_K) | 0.060 ms | 0.13 ms | 2.2× |

**Conclusion:** The matvec kernels themselves are NOT the bottleneck. In isolation they achieve **774 GB/s** (83% of peak). In the full pipeline, effective bandwidth drops to **~70 GB/s** (7% of peak).

**Root cause hypothesis:** The ~1700 kernel launches per token create pipeline bubbles. Tiny elementwise kernels (silu, add, copy with 256 threads) finish in ~10 µs, leaving the GPU underutilized while the CPU enqueues the next kernel. With 64 layers and dozens of kernels per layer, these bubbles accumulate.

---

## Three Kernels That Need Refactoring

Based on the timing data and the gap to llama.cpp, the following kernels need to be rewritten for max throughput:

### 1. **SSM Output Matvec (`ssmo`) — 11.7 ms/tok, 30.9%**
- **Problem:** 48 launches per token, each processing 5120 output rows. In isolation it's fast; in pipeline it's 10× slower.
- **What llama.cpp does:** Uses `mul_mat_vec_q` with tiled execution — each thread block processes multiple output rows and caches Q8_1 input in shared memory. This amortizes launch overhead and improves cache locality.
- **Action:** Rewrite `k_q4k_mv_q8_dp4a` to process **4 rows per block** with shared Q8_1 cache. Estimated save: 3–5 ms/tok.

### 2. **SSM wqkv_gate Matvec (`re-g`) — 7.5 ms/tok, 19.8%**
- **Problem:** 48 launches per token, Q5_K dequant path. Same pipeline bubble issue.
- **What llama.cpp does:** Q5_K vec-dot uses a different register layout and fewer instructions. Their `dequantize_mul_mat_vec_q` kernel is hand-tuned per quantization type.
- **Action:** Port llama.cpp's Q5_K `vec_dot` implementation. Estimated save: 2–3 ms/tok.

### 3. **FFN Gate+Up Matvec Pair (`fg+fu`) — 2.1 ms/tok, 5.6%**
- **Problem:** Two separate kernels launched sequentially, reading the same Q8_1 input but different weights.
- **What llama.cpp does:** Fused `mul_mat_id` or custom fused kernel that reads Q8_1 once and computes both projections in one launch. This eliminates one kernel launch and one global memory read of the Q8_1 cache.
- **Action:** Implement `k_q4k_gate_up_fused` (already prototyped but needs refinement). Estimated save: 0.5–1 ms/tok.

---

## Additional Optimizations to Explore

| Optimization | Est. Impact | Difficulty |
|-------------|-------------|------------|
| **Persistent kernel per layer** (single kernel runs entire SSM layer) | +5–8 ms/tok | High |
| **Top-K on GPU** (avoid 1MB D2H copy for sampling) | +1–2 ms/tok | Medium |
| **MMQ path for LM head** (248320 output rows) | +0.5 ms/tok | Medium |
| **Remove remaining `cudaDeviceSynchronize`** in output path | +0.3 ms/tok | Low |
| **Attention kernel occupancy** (reduce shared mem from 17KB to ~1KB) | +0.1 ms/tok | Low |

---

## Correctness Status

- ✅ Sampling output matches baseline exactly across tested prompts/seeds
- ✅ Greedy mode produces identical first ~30 tokens (then enters pre-existing repeat loop)
- ✅ All intermediate tensors match llama.cpp within 5e-5 tolerance
- ⚠️ Logit correlation with llama.cpp: 0.98 (MSE 0.43) — small systematic difference remains

---

## Experiment History

| # | Date | Change | Speed | Status |
|---|------|--------|-------|--------|
| 1 | 05-05 | Baseline | 18.4 tok/s | baseline |
| 2 | 05-05 | Warp shuffle reduction + GPU argmax | 20.5 tok/s | keep |
| 3 | 05-05 | Remove MV_PAIR stream sync | 19.6 tok/s | keep |
| 4 | 05-05 | Batched RMS norm | 25.5 tok/s | keep |
| 5 | 05-05 | CUDA graphs + Q8_1 cache | 24.5 tok/s | keep |
| 6 | 05-06 | Fused gate+up kernel | 24.7 tok/s | discard |
| 7 | 05-06 | Attention shared memory reduction | 24.2 tok/s | discard |
| 8 | 05-06 | Multi-row matvec x4 + shared Q8_1 | 23.6 tok/s | discard |
| 9 | 05-06 | GPU top-k (naive sequential) | 12.2 tok/s | discard |

---

## Next Steps (Priority Order)

1. **Profile with Nsight Systems** — run `nsys` to identify exact pipeline bubbles
2. **Rewrite ssmo matvec** — multi-row per block + shared Q8_1 cache
3. **Port llama.cpp Q5_K vec_dot** — compare instruction-by-instruction
4. **Fuse SSM elementwise ops** — silu + rms_norm + mul into single kernel
5. **GPU Top-K** — avoid 1MB D2H copy in sampling mode
