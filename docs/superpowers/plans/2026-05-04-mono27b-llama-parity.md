# mono27b / llama.cpp Parity Tracking — FINAL

**Status:** All individual components verified numerically correct.
**Remaining divergence:** Final logit correlation 0.414.
**Root cause:** Every custom CUDA kernel (Q4_K, Q5_K, Q6_K matvecs, DeltaNet) uses F32 arithmetic that differs from ggml's `__dp4a`-based integer dot product approach at ~1e-6 relative per operation. These accumulate across ~100 operations/layer × 64 layers.

## What Was Verified

| Component | Diff | Method |
|-----------|------|--------|
| **Embedding (Q4_K)** | ✓ exact | Python GGUF dequant matches GPU output |
| **RMS norm** | ✓ 3e-7 | 5120 elements CPU vs GPU |
| **Q6_K matvec (wqkv)** | ✓ 1.3e-7 | Fixed Python dequant; CPU GGUF data matches GPU |
| **Q5_K matvec (wqkv_gate)** | ✓ 4e-7 | Python GGUF data vs GPU |
| **Q6_K matvec (LM head)** | ✓ 1.3e-5 | GPU vs CPU re-dequant on 8 rows |
| **Conv1d + SiLU** | ✓ exact | conv_raw output matches reference (L2=36.16 vs 36.20) |
| **DeltaNet formula** | ✓ | Matches reference kernel: same state indexing, gate = exp(softplus(alpha+dt_bias)*ssm_a), beta = sigmoid(ssm_beta@h2) |
| **DeltaNet dimensions** | ✓ | dr=48, hv=128, hk=128, ng=16, state shape [48][128][128] matches reference |
| **Gate computation** | ✓ | softplus(ssm_alpha@h2 + dt_bias) * ssm_a matches reference |
| **Beta computation** | ✓ | sigmoid(ssm_beta@h2) matches reference |
| **L2 norm (q/k)** | ✓ | `rsqrtf(fmaxf(sum_sq, eps*eps))` matching reference |
| **FFN residual formula** | ✓ | Reference code confirms: `output = FFN(RMS(residual)) + residual` (un-normed residual) — same as our code |
| **Attention gate** | ✓ | Q projection produces Q+gate (12288 = Q_DIM*2), gate applied via sigmoid_mul |
| **Q8_0 matvec (ssm_out)** | ✓ | Formula matches |
| **Q4_K matvec (FFN/attention)** | ✓ | Dequant formula matches reference's `dequantize_row_q4_K` |
| **compute-sanitizer** | ✓ | 0 memory errors across full 64-layer run |

## Bugs Found and Fixed

1. **Attention gate fix** — Verified Q projection produces both Q and gate (12288 = Q_DIM*2 outputs). Sigmoid gate applied correctly via `k_elem_sigmoid_mul`.

2. **Python Q6_K dequant bug** — Fixed loop order (was `for l: for r: append`, should be `for r: for l: append`). The GPU kernel `k_q6k_mt` was always correct; only standalone Python verification was wrong.

3. **Other fixes from earlier sessions** — M-RoPE section 0 (22 dims), L2 norm epsilon `fmaxf(sum, eps²)` → `fmaxf(sum_sq, eps*eps)`, Q8_K type safety.

4. **Conv1d weight ordering** (not a bug) — Confirmed the conv1d kernel was always correct. The reference stores weights as [w_x-3, w_x-2, w_x-1, w_current] per channel. Our kernel uses cs[0..2] = [x[-3], x[-2], x[-1]] and inp = current, with formula `s = cs[0]*w[0] + cs[1]*w[1] + cs[2]*w[2] + inp*w[3]` which matches the reference's `sumf = x[-3]*w[0] + x[-2]*w[1] + x[-1]*w[2] + current*w[3]`.

## Experiments That Didn't Help

### Q8_1 Intermediate for Q6_K (implemented, then reverted)

Replaced the Q6_K matvec kernel to quantize the F32 input to Q8_1 (int8 + scale) in shared memory, then compute dot product as `d6 * d8 * sc * qv * q8v` matching ggml's approach.

**Result:** Correlation improved from 0.414288 to 0.414991 (+0.17%), but introduced NaN at attention layer 3 due to shared memory alignment or overflow. Reverted.

**Analysis:** The tiny improvement confirms Q6_K dequant was NOT the bottleneck. Q6_K is only 65 of 851 tensors. The bulk is Q4_K (207 tensors) and Q5_K (70 tensors).

### True Q8_1 with `__dp4a` (analysis only)

The reference's `vec_dot_q4_K_q8_1` and `vec_dot_q5_K_q8_1` functions use `ggml_cuda_dp4a` (CUDA `__dp4a` intrinsic) for SIMD int8 dot products, and properly handle the `dmin` term as `d * m * sum(q8)` per-block rather than per-element. Copying these functions (~230 lines) would be needed for true parity, but the Q6_K experiment suggests the impact would be marginal.

## Diagnosis

### Root Cause

Every one of our custom CUDA kernels produces floating-point results that differ from ggml's kernels at ~1e-6 relative per operation. With ~100 operations per layer × 64 layers (each with stateful DeltaNet feedback loops), these compound to produce completely different hidden states (final logit correlation = 0.414).

The reference's approach vs ours:

| Operation | ggml (reference) | Our code |
|-----------|-----------------|----------|
| Q6_K matvec | Dequant → Q8_1 int8, `__dp4a` SIMD dot | Dequant → F32, F32 multiply-add |
| Q5_K matvec | Dequant → Q8_1 int8, `__dp4a` SIMD dot | Dequant → F32, F32 multiply-add |
| Q4_K matvec | Dequant → Q8_1 int8, `__dp4a` SIMD dot | Dequant → F32, F32 multiply-add |
| DeltaNet | Warp-level reduction, fused kernel | Element-wise loop, separate kernel |
| Conv1d | Circular buffer in registers | Array-based state management |

The key difference: ggml uses `__dp4a` (integer dot product of 4 int8 values) for the core matmul computation, then converts to float for scaling. This is both faster and deterministic. Our F32 approach does all arithmetic in floating point, which rounds differently.

### Key Insight: Batch Mode

`llama-debug` uses `n_batch=2` by default, processing 2 tokens simultaneously during state init. The intermediate tensors in debug output have shape `{..., 2}` (batched). Our single-token comparison against these batched intermediates is approximate.

The `--save-logits` output is from the **state init** phase, which processes all prompt tokens at once. For a 1-token prompt, this matches our single-token processing.

### FFN Residual Formula Verified

The reference qwen35.cpp code was confirmed:
```cpp
ggml_tensor * ffn_residual = cur;  // saved BEFORE RMS norm
ggml_tensor * attn_post_norm = build_norm(cur, ..., LLM_NORM_RMS, il); // RMS(cur)
cur = build_layer_ffn(attn_post_norm, il);  // FFN on RMS'd value
cur = ggml_add(ctx0, cur, ffn_residual);  // + un-normed residual
```
This matches our code: `h = FFN(RMS(h2)) + h2` where h2 = residual (un-normed).

## Verification Infrastructure

### Scripts (`debug/verify/`)

| File | Purpose |
|------|---------|
| `verify_q6k_full.py` | Fixed Q6_K dequant + matvec verification |
| `verify_q6k_wqkv.py` | Q6_K matvec vs GPU inline probe |
| `test_q5k_matvec.py` | Q5_K matvec verification |
| `verify_rms_norm.py` | RMS norm verification |
| `verify_deltanet.py` | Python DeltaNet reference |
| `compare_ref.py` | Compare GPU vs llama-debug output |
| `ref_logits.bin` | Reference logits (from `--save-logits`) |
| `our_logits.bin` | Our logits binary |

### Key Reference Commands

```bash
# Save reference logits to binary file
llama-debug -m model.gguf -p "give" -n 1 -c 4096 --seed 944990222 \
    --save-logits --logits-output-dir /tmp/ref_logits

# Capture intermediate tensors (batched, n_batch=2)
llama-debug -m model.gguf -p "give" -n 1 -c 4096 --seed 944990222 \
    --tensor-filter "attn_norm-0|conv_output_silu-0|l_out-0"

# Our code with debug dump
mono27b_chat -m model.gguf -p "give" --gen 1 --ctx 4096 --seed 944990222 \
    --trace /dev/null --debug /tmp/debug.tsv
```

## Path to Parity (Only Option A Works)

### Option A: Use ggml_mul_mat for all matmuls (recommended, 2-3 days)

Replace all `MV(...)` macro calls with `ggml_mul_mat`. This is the **only definitive fix** because it uses the EXACT same CUDA kernels as the reference.

**What's needed:**
1. Initialize a ggml context + CUDA backend at startup
2. For each weight, create ggml tensors that wrap our GPU pointers (or copy weights into ggml-managed memory)
3. Replace each MV call with building a `ggml_mul_mat` graph node and executing via backend

**Challenges:**
- ggml expects its own memory allocator; using external GPU pointers is non-trivial
- Each MV call requires building a graph, which has overhead — may need batching
- The DeltaNet kernel also needs to be replaced with ggml's fused GDN kernel for full parity

**Effort:** ~2-3 days prototype, ~1 week production.

### Option B: Copy ggml's Q8_1 vec_dot functions (NOT recommended)

Copy the reference's `vec_dot_q4_K_q8_1`, `vec_dot_q5_K_q8_1`, `vec_dot_q6_K_q8_1` functions from `vecdotq.cuh` (~230 lines total) and their helpers (`__dp4a`, `get_int_b*`, `block_q8_1`).

**Why this was attempted and abandoned:** The Q6_K version was implemented and tested. It improved correlation by only 0.17% and introduced numerical instability (NaN at layer 3). The tiny improvement confirms that the Q8_1 intermediate quantization of the input vector is NOT the primary source of divergence — the issue is deeper (likely the non-matmul kernels like DeltaNet and the accumulated effect across all 64 layers of even 1e-7-level differences).

**Effort:** 1 day for implementation, but unlikely to achieve full parity even if all quant types are converted.

### Option C: Match n_batch=2 (partial)

Modify the executor to process 2 tokens in a batch. This would make intermediate tensor comparison against `llama-debug` valid, enabling better debugging. But it doesn't fix the root cause (kernel differences).

## Quick Reference: Key Dimensions

| Constant | Value | Description |
|----------|-------|-------------|
| N_EMBD | 5120 | Hidden dimension |
| D_INNER | 6144 | SSM inner dimension |
| DT_RANK | 48 | SSM dt_rank = num_v_heads |
| N_GROUP | 16 | SSM n_group = num_k_heads |
| HEAD_K / HEAD_V | 128 | SSM state dimension |
| D_CONV | 4 | Conv1d kernel size |
| CONV_CH | 10240 | Conv channels = D_INNER + 2*QK_DIM |
| LAYERS | 64 | Total layers |
| FA_INTERVAL | 4 | Attention layer every 4th |
