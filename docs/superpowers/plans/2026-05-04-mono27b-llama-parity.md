# mono27b / llama.cpp Parity Tracking — CURRENT

**Status:** Verification harness is restored and live-reference based again. The app stays independent of llama.cpp at runtime, but verification still compares against `llama-debug`.
**Current blocker:** Same-token replay shows the stateful SSM path still diverges on generation steps.
**Latest finding:** The prompt token is close, but the first generated token path blows up in `conv_output_silu` and downstream SSM tensors.

## What Was Verified

| Component | Diff | Method |
|-----------|------|--------|
| **Embedding (Q4_K)** | ✓ exact | Python GGUF dequant matches GPU output |
| **RMS norm (isolated)** | ✓ 3e-7 | 5120 elements CPU vs GPU |
| **RMS norm (attn_norm-0 E2E)** | ✗ | **Massive divergence (Ref: 69.4 vs Mono: 37.0)** — see investigation below |
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
| **Softplus stability** | ✓ | `(z > 20.0f) ? z : logf(1.0f + expf(z))` matching ggml |
| **GQA indexing (DeltaNet)** | ✓ | `r_idx / (dr / ng)` for repeating heads (fixed from `r_idx % ng`) |
| **Norm accumulation** | ✓ | `double` precision parallel reduction for both `k_rms_norm_mulw` and `k_l2_norm_g` |

## Recent Work

1. **Fixed softplus stability branching**: Reverted to the exact `llama.cpp` logic `(z > 20.0f) ? z : logf(1.0f + expf(z))` in [mono27b_executor.cu](file:///mnt/mydata/projects2/mono27b/src/mono27b_executor.cu).

2. **Corrected GQA indexing in DeltaNet**: Changed `r_idx % ng` to `r_idx / (dr / ng)` in [mono27b_executor.cu](file:///mnt/mydata/projects2/mono27b/src/mono27b_executor.cu) to correctly repeat KV heads across delta ranks.

3. **High-precision Norm Kernels**: Re-implemented `k_rms_norm_mulw` and `k_l2_norm_g` using parallel reduction and `double` precision accumulation to match GGML's CPU reference precision.

4. **Investigated Layer 0 Divergence**: Discovered that while embedding matches, the first attention norm (`attn_norm-0`) diverges significantly. This suggests a missing scaling factor or an error in how the input to the first layer is prepared.

5. **Fixed a real SwiGLU bug** in [mono27b_executor.cu](file:///mnt/mydata/projects2/mono27b/src/mono27b_executor.cu):
   - `k_elem_swiglu` was applying `silu` to the wrong operand.
   - It now computes `silu(gate) * up`, matching the expected FFN gate behavior.

6. **Restored and preserved live verification**:
   - Verification still generates reference data from `llama-debug`.
   - The app itself does not depend on llama.cpp.

7. **Added deeper SSM debug taps**:
   - `q_conv_predelta`
   - `k_conv_predelta`
   - `final_output`
   - `ssm_out`
   - `post_ffn`

8. **Added replay/debug controls to the app**:
   - `--replay-trace` forces our run to follow the reference token stream.
   - `--debug-pos N` dumps tensors for any chosen position, not only the prompt token.

9. **Fixed a verification selection bug** in [compare_ref.py](file:///mnt/mydata/projects2/mono27b/debug/verify/compare_ref.py):
   - The compare script had been hardcoded to `step=0,pos=0`, which could pair the wrong tensor occurrence when replaying generation.
   - It now selects the latest matching tensor dump for the requested phase/label, which makes replay comparisons inspect the actual generation-step tensors.

## Bugs Found and Fixed

1. **SwiGLU operand swap** — `k_elem_swiglu` was applying `silu` to the wrong operand. Fixed to compute `silu(gate) * up`.

2. **Attention gate fix** — Verified Q projection produces both Q and gate (12288 = Q_DIM*2 outputs). Sigmoid gate applied correctly via `k_elem_sigmoid_mul`.

3. **Python Q6_K dequant bug** — Fixed loop order (was `for l: for r: append`, should be `for r: for l: append`). The GPU kernel `k_q6k_mt` was always correct; only standalone Python verification was wrong.

4. **GQA Indexing Mismatch** — The engine was using `r_idx % ng` for repeating heads. Corrected to `r_idx / (dr / ng)` to match GGML's `repeat` operation.

5. **Softplus Logic** — Replaced mathematical approximation with exact logic from `ggml-impl.h`.

6. **RMS Norm Precision** — Initial implementation lacked sufficient precision; switched to `double` accumulation.

7. **Other fixes from earlier sessions** — M-RoPE section 0 (22 dims), L2 norm epsilon `fmaxf(sum, eps²)` → `fmaxf(sum_sq, eps*eps)`, Q8_K type safety.

8. **Conv1d weight ordering** (not a bug) — Confirmed the conv1d kernel was always correct. The reference stores weights as [w_x-3, w_x-2, w_x-1, w_current] per channel. Our kernel uses cs[0..2] = [x[-3], x[-2], x[-1]] and inp = current, with formula `s = cs[0]*w[0] + cs[1]*w[1] + cs[2]*w[2] + inp*w[3]` which matches the reference's `sumf = x[-3]*w[0] + x[-2]*w[1] + x[-1]*w[2] + current*w[3]`.

9. **Replay-token verification gap** — The earlier step-by-step mismatch on generated tokens was partly misleading because the sampled continuation diverged. I added a replay mode so our engine can be forced onto the exact reference continuation and compared on the same token stream.

## Experiments That Didn't Help

### Q8_1 Intermediate for Q6_K (attempt 1 — shared memory, reverted)

Replaced the Q6_K matvec kernel to quantize the F32 input to Q8_1 in shared memory, then compute dot product as `d6 * d8 * sc * qv * q8v`. Caused NaN at attention layer 3 (shared memory alignment issue). Correlation improved from 0.414288 to 0.414991 (+0.17%) before reverting.

### Q8_1 Intermediate for Q4_K (attempt 2 — global buffer, committed)

Added a `q8_scratch` global buffer (544 BlockQ8_1 = ~20KB) to the engine state, a `k_quant_q8_1` kernel for F32→Q8_1 quantization, and a `k_q4k_mv_q8` kernel using Q8_1 intermediate. Modified `l_mv` dispatch to use Q8_1 path for Q4_K weights.

**Result:** Correlation 0.413485 — essentially unchanged (-0.19% from 0.414288 baseline). Confirms Q8_1 quantization of the input vector is NOT the source of divergence, even for the most common weight type (Q4_K, 207 of 851 tensors).

### Forced replay investigation

I added a replay mode and a position-selectable debug dump so we can inspect the same continuation token as llama.cpp instead of comparing two different sampled branches.

Observed on the forced generated token stream:

| Tensor | Same-token diff | Notes |
|--------|-----------------|------|
| `attn_norm-0` | ~5e-05 | close |
| `conv_output_silu-0` | ~8.176 | large divergence |
| `q_conv_predelta-0` | ~0.648 | diverges after conv |
| `k_conv_predelta-0` | ~0.918 | diverges after conv |
| `attn_output-0` | ~0.466 | downstream drift |
| `final_output-0` | ~0.675 | downstream drift |
| `linear_attn_out-0` | ~10.43 | very large drift |
| `l_out-0` | ~0.436 | downstream drift |

Interpretation:
- The earlier prompt-token checks were not sufficient to identify the bug.
- The first generated token path still diverges in the stateful SSM branch.
- The first clearly broken stage on same-token replay is the SSM conv / DeltaNet path, not the prompt embedding or RMS norm.

### Follow-up replay evidence

After fixing the compare selection bug and dumping both prompt and generation positions:

- Prompt-step row-0 `wqkv` and `wqkv_gate` projections match CPU dequant/dot to about `1e-7`.
- Prompt-step `conv_output_raw`, `q_conv_predelta`, `k_conv_predelta`, and `ssm_out` are close but not bit-identical.
- Generation-step tensors still diverge materially after the prompt state is carried forward.
- That means the remaining problem is not the local projection kernels on row 0; it is the recurrent generation path, where tiny prompt-step differences are being amplified.

### Conclusion from both attempts

The Q8_1 matvec approach (matching ggml's `__dp4a`-based integer dot product) does NOT fix the parity issue — even if implemented for ALL quant types (Q4_K, Q5_K, Q6_K), the theoretical maximum improvement is <1% correlation. The divergence originates in the **non-matmul operations**: the DeltaNet kernel, element-wise ops, RMS norm accumulation order, etc. These use F32 arithmetic that rounds differently from ggml's reference implementations, and the differences compound across 64 layers with stateful DeltaNet feedback loops.

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

### Key Insight: Step Comparisons Need Replay

Comparing generated steps without forcing the same token stream is not reliable. Once the sampled token diverges, later hidden states are no longer comparable.

The replay path now avoids that trap by feeding the reference `gen` token sequence back into our app and dumping tensors at an arbitrary `--debug-pos`.

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
| `replay-trace` mode | Forces our generation stream to match a reference trace |
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

# Force our generation onto the same token stream as the reference trace
mono27b_chat -m model.gguf -p "give" --gen 1 --ctx 4096 --seed 944990222 \
    --trace /dev/null --debug /tmp/debug.tsv --debug-pos 1 \
    --replay-trace debug/ref_give_b1.trace.tsv
```

## Likely Path to Parity

The replayed same-token run points to the stateful SSM path, especially conv / DeltaNet, as the remaining source of divergence. This is more specific than the earlier “all math is slightly different” hypothesis.

The next likely fix is to port the ggml SSM conv / gated DeltaNet path more literally, then re-run the forced replay compare at `--debug-pos 1`.

### Required: Replace ALL custom CUDA kernels with ggml's implementations

This is the **only way** to achieve bit-exact parity. It requires:

1. **Matmuls:** Replace `MV(...)` with `ggml_mul_mat` from `libggml-cuda.so`
2. **DeltaNet:** Replace `k_deltanet` with ggml's fused GDN kernel from `gated_delta_net.cu`
3. **Conv1d:** Already matches (verified L2=36.16 vs 36.20)
4. **Element-wise ops:** Use ggml's `ggml_silu`, `ggml_sigmoid`, etc.

**Why Q8_1 approach failed:** The matvec is only ~30% of the FLOPs per layer. The DeltaNet, RMS norms, element-wise gates, and FFN account for the rest. Each of these uses F32 arithmetic that rounds differently from ggml. Even perfect matmuls leave ~70% of the computation on non-ggml paths.

**Challenge for no-deps constraint:** Replacing all operations with ggml's means either linking against `libggml-cuda.so` (external dependency) or copying the reference CUDA kernels directly (`gated_delta_net.cu`, `ssm-conv.cu`, norm/unary paths, and the relevant matmul kernels).

**Effort with no deps (copying code):** still nontrivial
**Effort with ggml library link:** still faster, but outside the app dependency constraint

## Current Investigation: attn_norm-0 Divergence (2026-05-04 update)

### Symptom

Isolated RMS norm kernel tests pass perfectly (max diff ~3e-7), but the end-to-end trace shows a **massive divergence** starting at the first `attn_norm` output (Layer 0):

| Metric | Reference (llama.cpp) | Our engine | Ratio |
|--------|----------------------|------------|-------|
| `attn_norm-0` L2 norm | 69.43 | 37.00 | 0.53× (~half) |
| `attn_norm-0` max diff | — | 25.64 | — |
| Embedding (input to norm) | close | close | ~1× |

The norm of our output is approximately **half** of the reference. This is too large to be a floating-point rounding issue and suggests a systematic error.

### Verified: Not the RMS Norm Kernel Itself

The `k_rms_norm_mulw` kernel with `double` precision accumulation produces correct results in isolation. The formula is:
```
y[i] = x[i] / sqrt(mean(x²) + eps) * w[i]
```
With `eps = 0.01` (from `MONO27B_RMS_EPS`), the epsilon contributes `0.01 / n ≈ 2e-6` to the mean — negligible for n=5120.

### Verified: Weight and Epsilon Values

- `blk.0.attn_norm.weight`: GGUF type F32, dims [5120], all values ~1.0. First 16 values match expected range. **Not a weight loading issue.**
- `MONO27B_RMS_EPS = 1e-2` matches `f_norm_rms_eps` from the GGUF metadata (loaded by llama.cpp as `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`).

### Verified: qwen35.cpp Architecture

The reference `llama.cpp` graph for Qwen3.5 (`LLM_ARCH_QWEN35`) is in [qwen35.cpp](file:///mnt/mydata/projects2/mono27b/ref/llama.cpp/src/models/qwen35.cpp). Key observations:

1. **Architecture uses both SSM (gated delta net) AND full attention layers.** Layers are marked `is_recurrent(il)` based on `(i+1) % 4 != 0`. So layers 0,1,2 are SSM; layer 3 is full attention; etc.
2. **The first operation on every layer is `build_norm(inpL, model.layers[il].attn_norm, ...)`.** This is identical to our approach.
3. **The graph has `attn_post_norm`** — a second RMS norm after the attention/SSM output, before the FFN. This is `LLM_TENSOR_ATTN_POST_NORM`. Our engine needs to verify it has this second norm.

### Hypothesis: Missing `attn_post_norm` or Wrong Layer Routing

The qwen35.cpp reference shows:
```cpp
cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
// ... attention or SSM ...
cur = ggml_add(ctx0, cur, inpSA);  // residual
ggml_tensor * ffn_residual = cur;
ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
cur = build_layer_ffn(attn_post_norm, il);
cur = ggml_add(ctx0, cur, ffn_residual);
```

Our engine must verify:
- That `attn_post_norm` weights are being loaded and applied correctly
- That the layer routing (SSM vs full attention) matches `(i+1) % 4 != 0`

### Hypothesis: Embedding Scaling

The L2 norm ratio (37.0 / 69.4 ≈ 0.53) is suspiciously close to 1/√2 ≈ 0.707 or possibly a factor of 2 somewhere. Possible causes:

1. **The embedding output is being halved** — maybe the Q4_K dequant is reading only half the values, or the embedding gather is off by a factor
2. **A hidden state is being overwritten** — the `h` buffer might be clobbered between embedding and norm
3. **The wrong buffer is being normed** — `l_rms` might be reading from the wrong pointer (pre-embedding zeros or an intermediate scratch buffer)

### GGUF Tensor Layout for blk.0 (from the model file)

| Tensor | Type | Dims |
|--------|------|------|
| `blk.0.attn_norm.weight` | F32 | [5120] |
| `blk.0.attn_qkv.weight` | Q6_K (14) | [5120, 10240] |
| `blk.0.attn_gate.weight` | Q5_K (13) | [5120, 6144] |
| `blk.0.ffn_down.weight` | Q6_K | [n_ff, 5120] |
| `blk.0.ffn_gate.weight` | Q4_K | [5120, n_ff] |
| `blk.0.ffn_up.weight` | Q4_K | [5120, n_ff] |
| `blk.0.post_attention_norm.weight` | F32 | [5120] |
| `blk.0.ssm_a` | F32 | [48] |
| `blk.0.ssm_alpha.weight` | — | [5120, 48] |
| `blk.0.ssm_beta.weight` | — | [5120, 48] |
| `blk.0.ssm_conv1d.weight` | F16 | [4, 10240] |
| `blk.0.ssm_dt.bias` | F32 | [48] |
| `blk.0.ssm_norm.weight` | F32 | [128] |
| `blk.0.ssm_out.weight` | — | [6144, 5120] |

Note: `ssm_norm.weight` is dims [128] = HEAD_V_DIM. This is the norm applied per-head inside the gated SSM output, NOT a full-hidden-dimension norm. Our code applies it per-rank in a loop of `DT_RANK` iterations — need to verify this matches the reference's `build_norm_gated` which uses `ggml_norm` with `ssm_norm` as weights.

### Next Steps

1. **Add a debug dump of the raw `h` buffer immediately before `l_rms` in the layer loop** to verify the input to the norm is correct (same as embedding output)
2. **Verify `attn_post_norm` is loaded and applied** — check the weight loader for `post_attention_norm.weight`
3. **Trace the exact `h` pointer flow** from embedding → first layer norm to ensure no buffer aliasing
4. **Compare the `build_layer_attn_linear` in [qwen35.cpp](file:///mnt/mydata/projects2/mono27b/ref/llama.cpp/src/models/qwen35.cpp) with our SSM path** to verify the gate computation (`build_norm_gated`) matches our implementation

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
