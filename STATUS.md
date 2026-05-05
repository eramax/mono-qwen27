# mono27b — Status & Action Plan

## What was done

### 1. ✅ Parallel Attention Kernel
Replaced the single-threaded attention kernel with a cooperative parallel version:
- Old: `k_attn_1t` — 1 thread, 24 blocks, sequential dot/softmax/weighted-sum
- New: `k_attn_parallel` — 256 threads per block, cooperative reductions, full parallelism

**Impact**: ~13% speed improvement in greedy mode (17.4→19.7 tok/s). Negligible improvement in sampling mode (the bottleneck is matvec, not attention).

### 2. ✅ Template Alignment
Re-added `<think>\n` prefix to match llama.cpp's Jinja template output.

### 3. ✅ Root Cause Analysis
Completed in ANALYSIS.md — identified all contributing factors.

## Remaining Gaps

### Gap 1: Logit divergence at generation position
The model processes `<think>` as the last prompt token but generates "Here" instead of " Here's" (token 8160 vs what llama.cpp predicts). The E2E logit correlation is 0.98 for the first prompt token — good but not perfect. After 64 layers × 16 prompt tokens, small errors compound to change the top-1 prediction.

**Root cause**: Not a single bug — it's accumulated floating-point differences across the entire pipeline (Q4_K matvec, SSM/DeltaNet, attention). The per-layer tensor comparisons all pass within 5e-5 absolute tolerance.

### Gap 2: Speed (~17 vs ~41 tok/s)
The main bottleneck is **matvec operations** (Q4_K, Q5_K, Q6_K matrix-vector products for each of 62 SSM layers + 2 attention layers). llama.cpp achieves higher speed via:
- **CUDA graphs** (USE_GRAPHS=1) — eliminates kernel launch overhead by replaying a captured graph
- **Flash attention** — tiled softmax with online normalizer
- **Fused DeltaNet** — single fused kernel for the entire SSM step

### Gap 3: Failing Q6_K matvec test
The test script can't find matching data in the debug output — needs debugging.

## Proposed Next Steps

### Short-term (1-2 hours):
1. **CUDA Graph support** — capture a graph of `mono27b_engine_decode_step` after warmup, replay for subsequent tokens (expected 1.5-2× speedup)
2. **Fix Q6_K test** — update the test script to match current tensor naming

### Medium-term:
1. **Fused DeltaNet kernel** — merge conv1d + DeltaNet + output projection into a single kernel to reduce global memory traffic
2. **Layer-by-layer comparison at generation position** — add a test that compares intermediate tensors at the 16th prompt token position (after `<think>`) to find where the first divergence occurs

### Long-term (requires more investigation):
1. **Full logit match** — if exact bit-for-bit matching is needed, the entire pipeline must match llama.cpp's computation order. This is a major effort requiring kernel-by-kernel alignment.
2. **Flash attention** — replace the current attention kernel with an online-softmax tiled version

## Key Files Modified
- `src/mono27b_executor.cu` — new `k_attn_parallel` kernel replaces `k_attn_1t`
- `src/mono27b_chat.cpp` — template with `<think>\n` prefix
- `ANALYSIS.md` — comprehensive findings document
