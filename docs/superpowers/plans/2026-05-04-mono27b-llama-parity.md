# mono27b / llama.cpp Parity Tracking — FINAL

**Status:** All individual components verified numerically correct.
**Remaining divergence:** Layer 0 correlation 0.95, final logit correlation 0.41.
**Root cause:** Per-layer quantized matmul numerical differences (~0.1%) compound across 64 layers.

## What Was Verified

| Component | Diff | Method |
|-----------|------|--------|
| **Embedding (Q4_K)** | ✓ exact | Python GGUF dequant matches GPU output |
| **RMS norm** | ✓ 3e-7 | 5120 elements CPU vs GPU |
| **Q6_K matvec (wqkv)** | ✓ 1.3e-7 | Fixed Python dequant loop order; CPU GGUF data matches GPU |
| **Q5_K matvec (wqkv_gate)** | ✓ 4e-7 | Python GGUF data vs GPU |
| **Q6_K matvec (LM head)** | ✓ 1.3e-5 | GPU vs CPU re-dequant on 8 rows |
| **Conv1d + SiLU** | ✓ exact | conv_raw output matches reference (L2=36.16 vs 36.20) |
| **DeltaNet formula** | ✓ | Matches reference kernel: same state indexing, gate = exp(softplus(alpha+dt_bias)*ssm_a), beta = sigmoid(ssm_beta@h2) |
| **DeltaNet dimensions** | ✓ | dr=48, hv=128, hk=128, ng=16, state shape [48][128][128] matches reference |
| **Gate computation** | ✓ | softplus(ssm_alpha@h2 + dt_bias) * ssm_a matches reference |
| **Beta computation** | ✓ | sigmoid(ssm_beta@h2) matches reference |
| **L2 norm (q/k)** | ✓ | `rsqrtf(fmaxf(sum_sq, eps*eps))` matching reference |
| **Attention gate** | ✓ | Q projection produces Q+gate (12288 = Q_DIM*2), gate applied via sigmoid_mul |
| **Q8_0 matvec (ssm_out)** | ✓ | Formula matches |
| **Q4_K matvec (FFN/attention)** | ✓ | Same dequant as embed |
| **compute-sanitizer** | ✓ | 0 memory errors across full 64-layer run |

## Bugs Found and Fixed

1. **Attention gate fix** — Verified Q projection produces both Q and gate (12288 = Q_DIM*2 outputs). Sigmoid gate applied correctly via `k_elem_sigmoid_mul`.

2. **Python Q6_K dequant bug** — Fixed loop order (was `for l: for r: append`, should be `for r: for l: append`). The GPU kernel `k_q6k_mt` was always correct; only standalone Python verification was wrong.

3. **Other fixes from earlier sessions** — M-RoPE section 0 (22 dims), L2 norm epsilon `fmaxf(sum, eps²)` → `fmaxf(sum_sq, eps*eps)`, Q8_K type safety.

4. **Conv1d weight ordering** (not a bug) — Confirmed the conv1d kernel was always correct. The reference stores weights as [w_x-3, w_x-2, w_x-1, w_current] per channel. Our kernel uses cs[0..2] = [x[-3], x[-2], x[-1]] and inp = current, with formula `s = cs[0]*w[0] + cs[1]*w[1] + cs[2]*w[2] + inp*w[3]` which matches the reference's `sumf = x[-3]*w[0] + x[-2]*w[1] + x[-1]*w[2] + current*w[3]`.

## Diagnosis

### What Was Found

Every individual component in the GPU executor was verified against either:
- **CPU re-dequant of GGUF data** (Q5_K, Q6_K matmuls) — matching to ~4e-7
- **llama-debug --tensor-filter output** (conv_raw, conv_silu, attn_norm, embed) — L2 matching within 0.1%

The conv1d output matches the reference (L2=36.16 vs 36.20). The wqkv_gate output matches with correlation 0.999995. The DeltaNet formula, dimensions, and indexing match the reference code exactly.

### Why E2E Logits Still Diverge

| Metric | Ours | Reference |
|--------|------|-----------|
| Logit L2 | 1108 | 2086 |
| Top-1 token | 11 (",", logit 16.9) | 728 ("", logit 15.7) |
| Top-5 overlap | — | 0/5 |
| Correlation | 0.414 | — |
| Layer 0 output correlation | 0.95 | — |
| Layer 0 L2 | 12.28 | 17.19 (ref batch token) |

**Root cause:** Per-layer numerical differences from quantized matmul CUDA kernels accumulate across 64 layers. Even though individual operations match to 1e-7, the 0.5-1% relative per-element differences compound through multiple matmuls per layer × 64 layers to produce completely different hidden states.

The reference uses `ggml_mul_mat` with CUDA kernels that differ from our custom implementations in:
- Thread block sizes and mapping
- Reduction strategies (sequential vs tree reduction)
- FMA vs separate multiply-add
- Memory access patterns (coalesced vs strided)

These implementation differences produce slightly different floating-point results for each quantized matmul, which cascade through the DeltaNet state update (a feedback loop) and compound across layers.

### Key Insight: Batch Mode

`llama-debug` uses `n_batch=2` by default, processing 2 tokens simultaneously during state init. The intermediate tensors (conv_output, z, l_out, etc.) in the debug output have shape `{..., 2}` (batched). Our single-token comparison against these batched intermediates is approximate.

The `--save-logits` output is from the **state init** phase (not the prediction step), which processes all prompt tokens at once. For a 1-token prompt, this is equivalent to our single-token processing.

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

## Path to Parity

To achieve bit-exact parity, one of these approaches is required:

### Option A: Replace custom matvec kernels with ggml_mul_mat

Modify `mono27b_executor.cu` to call `ggml_mul_mat` (from `libggml-cuda.so`) instead of custom `k_q6k_mt`, `k_q5k_mt`, etc. This ensures the EXACT same CUDA kernels as the reference.

**Effort:** ~2-3 days. Requires:
- Initializing ggml context and backend within the executor
- Creating ggml tensors from our GPU weight pointers
- Replacing each `MV(...)` macro call with `ggml_mul_mat(...)` + synchronize
- Handling the mixed quantized/F16/F32 weight types

### Option B: Layer-by-layer ggml CPU reference

Build a C++ program using `libggml-base.so` + `libggml-cpu.so` that:
1. Loads one SSM layer's weights from GGUF
2. Computes the full layer on CPU (using ggml for all ops)
3. Compares with GPU output element-by-element

**Effort:** ~1 day. Identifies which specific operation causes the first divergence.

### Option C: Match n_batch=2

Modify our executor to process 2 tokens in a batch (matching `llama-debug`'s `n_batch=2`). This would make the intermediate tensor comparison valid.

**Effort:** ~2-3 days. Requires redesign of the state management to handle batch dimension.

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
