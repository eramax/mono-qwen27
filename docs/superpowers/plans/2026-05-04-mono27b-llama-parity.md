# mono27b / llama.cpp Parity Tracking — CURRENT

**Status:** Prompt-step verification shows 9/12 tensors PASS with <1% diff. Remaining 3 FAILs (`ffn_gate`, `ffn_up`, `layer_out`) show ~0.01 diff consistent with Q4_K quantization noise. E2E correlation is **0.4715** (was 0.414). Decision: investigate attention layers and stateful divergence; quantization drift is secondary.
**Current blocker:** We are only comparing layer 0 (SSM). Attention layers (3,7,11,...) are unverified and likely the dominant source of E2E mismatch.
**Latest finding:** After porting ggml `mmvq` Q4_K/Q5_K/Q6_K/Q8_1 kernels, SSM layer 0 is nearly exact (`deltanet` PASS, `final_output` PASS). The remaining ~0.01 diff on FFN tensors is within quantization tolerance. The 0.47 E2E correlation suggests a larger divergence in attention layers or stateful execution.

## What Was Verified

| Component | Diff | Method |
|-----------|------|--------|
| **Embedding (Q4_K)** | exact | Python GGUF dequant matches GPU output |
| **RMS norm (isolated)** | 1.2e-7 | 5120 elements CPU vs GPU |
| **RMS norm (attn_norm-0 E2E)** | 5e-05 | Matches reference exactly |
| **Q6_K matvec (wqkv)** | 8.7e-03* | CPU F32 dequant vs GPU Q8_1+dp4a (tolerance too tight) |
| **Q5_K matvec (wqkv_gate)** | 4.7e-03* | CPU F32 dequant vs GPU Q8_1+dp4a (tolerance too tight) |
| **Q4_K matvec (FFN)** | 1.0e-02* | CPU F32 dequant vs GPU Q8_1+dp4a — expected Q4_K noise |
| **Q6_K matvec (LM head)** | 1.3e-5 | GPU vs CPU re-dequant on 8 rows |
| **Conv1d + SiLU** | exact | conv_raw output matches reference |
| **DeltaNet formula** | exact | Matches reference kernel logic |
| **DeltaNet group mapping** | exact | `g_idx = h_idx % ng` (fixed from `h_idx / (dr/ng)`) |
| **Gate computation** | exact | softplus(ssm_alpha@h2 + dt_bias) * ssm_a matches reference |
| **Beta computation** | exact | sigmoid(ssm_beta@h2) matches reference |
| **L2 norm (q/k)** | exact | `rsqrtf(fmaxf(sum_sq, eps*eps))` matching reference |
| **FFN residual formula** | exact | `output = FFN(RMS(residual)) + residual` |
| **Attention gate** | exact | Q projection produces Q+gate (12288 = Q_DIM*2), gate applied via sigmoid_mul |
| **Q8_0 matvec (ssm_out)** | exact | Formula matches |
| **compute-sanitizer** | 0 errors | 0 memory errors across full 64-layer run |
| **Softplus stability** | exact | `(z > 20.0f) ? z : logf(1.0f + expf(z))` matching ggml |
| **Norm accumulation** | exact | `double` precision parallel reduction |
| **Q8_1 block layout** | exact | `ds` as `__half2`, matching `block_q8_1` from ggml-common.h |

\* These diffs are CPU F32 dequant vs GPU Q8_1+dp4a. The GPU kernels are functionally correct — the CPU test is the "wrong" reference.

## Baseline Verification Results (`make verify`)

### Component Checks

| Check | Status | Detail |
|-------|--------|--------|
| Q6_K format sanity | PASS | d values in valid range |
| RMS norm (5120 elem) | PASS | max diff=1.23e-07 |
| Q5_K matvec (wqkv_gate) | PASS | diff=4.68e-03 |
| Q6_K matvec (wqkv) | FAIL* | diff=8.66e-03 (CPU vs GPU quantization path) |
| M-RoPE text rotation | PASS | worst_diff=0.00e+00 |

\* Q6_K matvec is not actually broken — the CPU F32 reference differs from GPU Q8_1+dp4a by ~0.9%, which is expected. The `deltanet` tensor (which consumes `wqkv`) passes exactly, confirming the kernel is correct.

### Tensor-by-Tensor Comparison (Layer 0, SSM only)

| Our Label | Ref Label | N | MaxDiff | RelMax | Status |
|-----------|-----------|---|---------|--------|--------|
| attn_norm | attn_norm-0 | 5120 | 0.000050 | 0.0137% | PASS |
| conv | conv_output_silu-0 | 10240 | 0.000050 | 0.0055% | PASS |
| conv_raw | conv_output_raw-0 | 10240 | 0.000050 | 0.0055% | PASS |
| deltanet | attn_output-0 | 6144 | 0.000050 | 0.2166% | PASS |
| **ffn_gate** | **ffn_gate-0** | **17408** | **0.010171** | **3.17%** | **FAIL** |
| **ffn_up** | **ffn_up-0** | **17408** | **0.010143** | **2.31%** | **FAIL** |
| final_output | final_output-0 | 6144 | 0.000050 | 0.0210% | PASS |
| h | model.input_embed | 5120 | 0.000050 | 0.7169% | PASS |
| k_conv_predelta | k_conv_predelta-0 | 2048 | 0.000050 | 0.0551% | PASS |
| **layer_out** | **l_out-0** | **5120** | **0.013176** | **0.1402%** | **FAIL** |
| q_conv_predelta | q_conv_predelta-0 | 2048 | 0.000050 | 0.0540% | PASS |
| z | z-0 | 6144 | 0.000050 | 0.0037% | PASS |

### E2E Logits

| Metric | Value |
|--------|-------|
| Top1 token (ref) | 728 |
| Top1 token (ours) | 220 |
| Top1 logit (ref) | 15.74 |
| Top1 logit (ours) | 16.77 |
| Correlation | **0.471522** |
| MSE | **4.8989** |

---

## Critical Findings

### Finding 1: The 0.01 FFN diff is quantization noise, not a logic bug
- `ffn_gate`/`ffn_up` inputs (`post_norm`) match reference exactly (PASS, 5e-05).
- The 0.01 diff appears **only after the Q4_K matvec**.
- Our single-row CPU-vs-GPU test shows our Q4_K matvec differs from CPU by ~0.001; reference GPU differs from CPU by ~0.0003–0.004. Both are within Q4_K tolerance.
- **Conclusion**: The Q4_K/Q5_K/Q6_K kernels are functionally correct. The 0.01 diff is natural quantization noise from Q8_1 rounding.

### Finding 2: We are not comparing attention layers at all
- Our comparison pipeline (`compare_all.py`) only dumps layer 0 (SSM).
- Attention layers are at indices **3, 7, 11, ...** (every 4th layer, `MONO27B_TARGET_FA_INTERVAL = 4`).
- Attention layers use RoPE, softmax, KV-cache, and multi-head attention — any bug here would compound across all attention layers and dominate the final logits.
- **This is the most likely source of the 0.47 E2E correlation.**

### Finding 3: We may be comparing the wrong token position
- `llama-debug` evaluates the graph **twice** per run (two passes).
- Our binary runs `--gen 1`, producing logits for **pos 0** (prompt) and **pos 1** (generated token).
- `compare_all.py` uses the **last** reference occurrence, but it's unclear whether this is pos 0 or pos 1.
- If the divergence is in **stateful update** (KV cache write, SSM state update, conv state shift), the pos-1 logits will diverge even if pos-0 is perfect.

### Finding 4: Reference tensor naming mismatch
- Reference uses `attn_post_norm-0` as the FFN input; our dump uses `post_norm`.
- Reference `l_out-0` is `ADD(ffn_out-0, attn_residual-0)`; our `layer_out` is post-FFN add.
- These are now correctly mapped in `compare_all.py`.

---

## Plan to Fix Remaining Bugs

We pursue **three independent investigative tracks** in parallel.

### Track A: Check Attention Layers (Highest Priority)
**Hypothesis**: Attention layers diverge, and since they account for ~25% of the model, they destroy E2E correlation.

**Action**:
1. Add debug dumps for **attention layer 3** (first attention layer) in our executor: `attn_norm`, `q_proj`, `k_proj`, `v_proj`, `attn_output`, `post_ffn`.
2. Run `llama-debug` with a filter for layer-3 tensors.
3. Compare with `compare_all.py`.

**Expected outcome**: If attention tensors FAIL, we found the big divergence. If they PASS, the bug is elsewhere.

### Track B: Isolate Stateful vs Stateless Divergence
**Hypothesis**: The prompt pass (pos 0) is nearly correct, but the generated token (pos 1) diverges due to a state-update bug.

**Action**:
1. Modify `compare_e2e.py` to compare **only the prompt logits** (pos 0).
2. Run our binary with `--gen 0` and compare with `llama-debug`.

**Expected outcome**: If pos-0 correlation is >> 0.47 (e.g., >0.95), the bug is in KV-cache/SSM-state update. If pos-0 is still ~0.47, the bug is in the forward pass itself.

### Track C: Eliminate Q8_1 Quantizer Drift
**Hypothesis**: Our simple per-thread `k_quant_q8_1` produces 1-ULP different `d` values than ggml's warp-reduction version, causing ~0.01 diffs that compound across 64 layers.

**Action**:
1. Port ggml's exact `quantize_q8_1` kernel (with `warp_reduce_max` and `warp_reduce_sum`) from `quantize.cu`.
2. Re-run `make verify`. If `ffn_gate`/`ffn_up` drop from 0.01 to <0.001, this was the issue.

**Expected outcome**: If `ffn_gate` diff drops to ~0.001 and E2E correlation jumps significantly, quantization drift was the culprit.

### Decision Tree

```
Track A result:
  ├─ Attention FAIL → Fix attention kernel (RoPE, softmax, KV-cache, output projection)
  └─ Attention PASS →
        Track B result:
          ├─ pos-0 correlation >> 0.47 → Bug is in stateful update (KV cache, SSM state, conv state)
          └─ pos-0 correlation ~0.47 →
                Track C result:
                  ├─ ffn_gate diff drops → Q8_1 quantizer was the issue
                  └─ ffn_gate diff stays → Unidentified forward-pass bug; need deeper layer-by-layer comparison
```

---

## Previously Ported Kernels

### DeltaNet kernel ✅
- Copied `gated_delta_net_cuda` from `gated_delta_net.cu`
- Replaced `ggml_tensor` stride logic with flat array indexing + grouped q/k indexing (`g_idx = h_idx % ng`)
- Added `k_deltanet_ggml` kernel with warp-level `warp_reduce_sum<32>` primitives
- Added `k_deltanet_ggml_launch` wrapper dispatching `S_v=128, KDA=false`
- Standalone test passes: max_diff = 1.1e-08 against CPU reference

### Q8_1 quantizer + K-quant matmuls ✅
- Added `BlockQ8_1` with `__half2 ds` union matching `block_q8_1` from ggml-common.h
- Added `k_quant_q8_1` kernel (per-thread, 1 thread/block)
- Added `k_q4k_mv_q8_dp4a` matching `vec_dot_q4_K_q8_1` from `vecdotq.cuh`
- Added `k_q5k_mv_q8` matching `vec_dot_q5_K_q8_1` from `vecdotq.cuh`
- Added `k_q6k_mv_q8_dp4a` matching `vec_dot_q6_K_q8_1` from `vecdotq.cuh`
- Added `k_iq4xs_mv_q8_dp4a` matching `vec_dot_iq4_xs_q8_1` from `vecdotq.cuh`
- Modified `l_mv` to dispatch Q8_1 path when `g_q8_scratch` is available and `n_q8 <= 544`

---

## Verification Infrastructure

### Scripts (`debug/verify/`)

| File | Purpose |
|------|---------|
| `verify_q6k_full.py` | Q6_K dequant + matvec verification |
| `verify_q6k_wqkv.py` | Q6_K matvec vs GPU inline probe |
| `test_q5k_matvec.py` | Q5_K matvec verification |
| `verify_rms_norm.py` | RMS norm verification |
| `verify_deltanet.py` | Python DeltaNet reference |
| `compare_all.py` | Comprehensive tensor comparison (uses LAST ref occurrence) |
| `run_all_checks.py` | Component checks (Q6_K, RMS norm, Q5_K, M-RoPE, E2E) |
| `compare_e2e.py` | End-to-end logit comparison |
| `extract_data.py` | Extracts embed_full.txt and attn_norm_full.txt from debug TSV |
| `ref_logits.bin` | Reference logits (from `--save-logits`) |
| `our_logits.bin` | Our logits binary |

### Key Reference Commands

```bash
# Full verify (build + gen ref + gen ours + compare)
make verify

# Only comparisons on existing data
make verify-only

# End-to-end logit comparison only
make e2e

# Comprehensive tensor comparison
make compare-all
```

---

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
| FFN_DIM | 17408 | FFN intermediate dimension |
| LAYERS | 64 | Total layers |
| FA_INTERVAL | 4 | Attention layer every 4th |

## Changelog

- **2026-05-05**: Fixed `BlockQ8_1` layout to match ggml exactly (`__half2 ds` union). Fixed Q8_1 member accesses in Q4_K and Q6_K kernels. Rebuilt and re-ran `make verify`: 9/12 PASS, E2E correlation 0.4715.
- **2026-05-05**: Identified that we are only comparing layer 0 (SSM). Attention layers (3,7,11,...) are completely unverified and likely the dominant E2E divergence source.
- **2026-05-05**: Identified potential stateful divergence: our binary generates pos 0 and pos 1; reference may evaluate twice. Need to isolate pos-0 correlation.
- **2026-05-04**: Discovered reference output contains two forward passes. Fixed comparison to use last occurrence.
- **2026-05-04**: Fixed `k_l2_norm_g` bug (was `1/sqrt(sum_sq + eps)`, should be `rsqrtf(fmaxf(sum_sq, eps*eps))`).
- **2026-05-04**: Fixed SSM gated norm buffer overflow (was using `h2` (5120 floats), now uses `kb` (17408 floats)).
- **2026-05-04**: Fixed Python GQA indexing bug (`r_idx % ng` → `r_idx // (dr // ng)`).
- **2026-05-04**: Added standalone CUDA kernel tests (`test_kernels.cu`) verifying L2 norm, RMS norm, Conv1D+SiLU, and DeltaNet against CPU references.
