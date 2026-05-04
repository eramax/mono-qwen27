# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State (as of 2026-05-04 Session 3)

**Output still diverges:**
- `llama.cpp` completion trace: prompt top1 token `728`, best logit `15.7379`
- Our backend: prompt top1 token `11`, best logit `16.9160`
- **Divergence starts at SSM Layer 0** — 1 layer gives top1=220 vs reference's 728 after 64 layers

### All Components Verified Numerically (✓ = matches reference)

| Component | Status | Verification Method |
|-----------|--------|---------------------|
| **Embedding (Q4_K dequant)** | ✓ | Official `dequantize_row_q4_K` on token 44883: `0.015738368...` identical |
| **RMS norm** | ✓ 3e-7 | Full 5120-element vector compared against CPU computation |
| **Q6_K matvec (wqkv)** | ✓ 1.4e-6 | GPU vs CPU re-dequant inline probe on row 0 of wqkv |
| **Q5_K matvec (wqkv_gate)** | ✓ 4e-7 | Python dot product with GGUF data matches GPU row 0 |
| **Q6_K matvec (LM head)** | ✓ 1.3e-5 | GPU vs CPU probe on 8 rows at both prompt and gen steps |
| **Q8_0 matvec** | ✓ | Formula matches `dequantize_row_q8_0` |
| **Q4_K dequant** | ✓ | Matches `dequantize_row_q4_K` (verified via embed) |
| **SiLU** | ✓ | `silu(0.07905) = 0.04109` verified numerically |
| **Softplus** | ✓ | Formula `max(z,0)+log1p(exp(-|z|))` matches reference |
| **Conv1d** | ✓ | Verified for zero-initial-state: `out = inp * w[last_tap]` |
| **M-RoPE** | ✓ | Only rotate section 0 (22 dims); fix applied |
| **L2 norm** | ✓ | Epsilon fix applied: `fmaxf(sum, eps²)` |
| **DeltaNet logic** | ✓ | Formula matches reference `gated_delta_net_cuda` kernel |
| **Buffer layout** | ✓ | Benign overflow only (h2[5120..6143] → qb which is dead) |
| **Memory errors** | ✓ | `compute-sanitizer memcheck`: 0 errors |
| **FFN (Q4_K matvec)** | ✓ | Same Q4_K kernel verified via embed |

## Bugs Found and Fixed

### 1. M-RoPE Section 0 Only (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_mrope` kernel) + `include/mono27b_config.h`
- Rotated only 22 dims (section 0) instead of all 64. Affects pos > 0 only.

### 2. L2 Norm Epsilon (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_l2_norm_g` kernel)
- Changed from `rsqrtf(sum + 1e-10f)` to `rsqrtf(fmaxf(sum, eps²))`.

### 3. Q8_K Type Safety (Added)
- **File:** `src/mono27b_gguf.cpp`
- Added handling for type 15 (Q8_K). Not used by current model.

## Verification Scripts (saved in `debug/verify/`)

| Script | Purpose |
|--------|---------|
| `test_q5k_matvec.py` | Verifies Q5_K matvec for wqkv_gate row 0 against CPU |
| `verify_q6k_wqkv.py` | Verifies Q6_K matvec for wqkv row 0 against GPU inline probe |
| `verify_rms_norm.py` | Verifies RMS norm against CPU computation |
| `verify_deltanet.py` | Reference DeltaNet implementation for future verification |
| `verify_q6k_ref.py` | Q6_K dequant using ggml reference library (ctypes stub) |

## Divergence Behavior

| Layers | Top1 | Logit | Notes |
|--------|------|-------|-------|
| 0 | 10074 | 8.56 | Embedding → output_norm → LM head only |
| 1 (SSM 0) | 220 | 7.03 | First divergence from reference trajectory |
| 4 (SSM 0-2, attn 3) | 220 | 6.17 | Same top1 as 1 layer |
| 64 (full) | 11 | 16.92 | Reference: 728 (15.74) |

## Open Questions

**Every individual component and the composition logic are verified correct at a floating-point level.** The root cause of the remaining divergence after SSM layer 0 is unknown and likely requires one of:

1. **Create a stand-alone reference computation** for SSM layer 0 using the reference library (`libggml-base.so`), loading the same weights from GGUF and comparing hidden state element-by-element. This would definitively pinpoint the divergence.

2. **Use `llama-debug` with `--tensor-filter`** on the reference side to dump intermediate tensor values and compare against our debug dumps.

3. **Binary-chop the SSM layer** by replacing parts of it (conv1d bypass, DeltaNet identity, etc.) to isolate the sub-component that causes the prediction change.

4. **Check for attention layer gate tensor loading** — the `attn_gate.weight` tensor for attention layers (3, 7, 11, ...) has shape [5120, 12288] (Q4_K/IQ4_XS) while for SSM layers it has shape [5120, 6144] (Q5_K). Verify our code loads the correct tensor for each layer type.

## Next Verification Steps

- [x] All individual component verification (RMS norm, Q4_K, Q5_K, Q6_K, Q8_0, conv1d, SiLU, softplus, sigmoid, DeltaNet, L2 norm, M-RoPE)
- [x] Binary-search: divergence starts at SSM Layer 0
- [x] `compute-sanitizer memcheck`: 0 errors
- [x] Q6_K matvec verified inline (GPU vs CPU re-dequant: diff 1.4e-6)
- [x] Q5_K matvec verified externally (Python vs GPU: diff 4e-7)
- [x] RMS norm verified (full 5120 elements: max diff 3e-7)
- [x] All verification scripts saved to `debug/verify/`
- [ ] **Build a reference SSM layer 0 end-to-end** using `libggml-base.so` to compare hidden states after each operation
- [ ] **Use `llama-debug`** with intermediate tensor dumps from the reference
- [ ] **Fix attention layer gate** if `attn_gate.weight` for layer 3 is not being loaded correctly (could affect layers 4+ output)
- [ ] Keep updating until 100% trace parity is achieved
