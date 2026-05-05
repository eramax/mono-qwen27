# mono27b — Speed Optimization Results

## Changes Made

### 1. Parallel Attention Kernel (k_attn_1t → k_attn_parallel)
- **Before**: Single-threaded attention — 1 thread did dot product, softmax, and weighted-sum for all 256 head dimensions
- **After**: 256 threads cooperate via tree reduction in shared memory
- **Impact**: 13% speedup in greedy mode

### 2. Removed Per-Operation GPU Synchronizations
- **Before**: ~640 `cudaDeviceSynchronize()` calls per token (10 per layer × 64 layers)
- **Before**: ~496 `check_finite_device()` calls per token (8 per SSM layer × 62 layers)
- **After**: Single sync at end of decode step
- **Impact**: +6.5% overall speed (17.0 → 18.1 tok/s)

### 3. MV Refactoring (infrastructure)
- Split `l_mv()` into `l_quant_q8()`, `l_mv_q8()`, `l_mv_fallback()`
- Enables future optimization: quantize input once, reuse for multiple matvecs

### 4. CUDA Graphs (attempted, disabled)
- KV cache write position (`pos`) changes each step, making graph replay invalid
- Would need kernel modifications to read `pos` from device memory
- llama.cpp handles this with dynamic graph re-capture per step

## Current Bottlenecks

| Component | Est. Time/Token | % of Total |
|-----------|----------------|------------|
| Q4_K matvec (5 per SSM layer × 62 + 3 per attn × 2 = 316 total) | ~40 ms | 72% |
| Attention (2 layers) | ~3 ms | 5% |
| LM head (1 Q6_K matvec) | ~4 ms | 7% |
| Element-wise ops (norms, activations, etc.) | ~5 ms | 9% |
| Kernel launch overhead | ~3 ms | 5% |

**Total: ~55 ms/token = 18.1 tok/s**

## Next Steps

### What's Done:
1. ✅ **Parallel attention kernel** — cooperative 256-thread dot/softmax/weighted-sum
2. ✅ **Removed 1136 syncs per token** — 640 cudaDeviceSynchronize + 496 check_finite_device eliminated
3. ✅ **Concurrent matvec pairs** — quantize shared input once, launch ffn_gate+ffn_up and wqkv+wqkv_gate concurrently via CUDA streams
4. ✅ **MV refactoring** — clean split of quantize/matvec/fallback paths

**Speed: 17.0 → 18.3 tok/s (+7.6%)**
**Output: Correct, unchanged from baseline**

### Remaining Bottlenecks:
1. Q4_K matvec achieves ~25% of RTX 3090 bandwidth — improving this is the path to 35+ tok/s
2. CUDA graphs blocked by KV cache `pos` parameter (would need kernel changes)

### Future Work:
1. **Optimize Q4_K matvec memory access** — improve coalescing for better bandwidth utilization
2. **Fix Q6_K matvec test** — update test script
3. **Fix BPE tokenizer** — verify exact match with llama.cpp
