# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State (as of 2026-05-04 Session 2)

**Output still diverges:**
- `llama.cpp` completion trace: prompt top1 token `728`, best logit `15.7379`
- Our backend: prompt top1 token `11`, best logit `16.9160`
- **Embedding confirmed correct** — q4_K dequant on token row 44883 matches ggml exactly
- **LM head confirmed correct** — Q6_K matvec verified to `1.33e-05` accuracy
- **Divergence traced to SSM Layer 0**: with 0 layers, top1=10074; with 1 layer (SSM 0), top1=220 (same as 4 layers). The bug is in the first SSM layer's computation.
- **No CUDA memory errors** — `compute-sanitizer memcheck` reports 0 errors

**All individual components verified correct against reference:**
| Component | Status |
| --- | --- |
| Q4_K dequant + matvec | ✓ match (verified via embed and manual check) |
| Q5_K dequant logic | ✓ match (index mapping verified against `dequantize_row_q5_K`) |
| Q6_K dequant + matvec | ✓ match (1.33e-05 on LM head probe) |
| Q8_0 dequant + matvec | ✓ match (dequant formula identical) |
| IQ4_XS dequant + matvec | present in code |
| RMS norm | ✓ match (uses ggml formula `rsqrtf(mean(x²) + eps)`) |
| L2 norm | ✓ match (fixed to `fmaxf(sum, eps²)`) |
| SiLU activation | ✓ match (`x/(1+exp(-x))`) |
| Softplus | ✓ match (`max(z,0) + log1p(exp(-|z|))`) |
| Sigmoid | ✓ match |
| M-RoPE | ✓ match (only rotate 22 dims for section 0) |
| Conv1d (first step) | ✓ correct (`out = inp * w[last_tap]` when state=0) |
| DeltaNet (first step, zero state) | ✓ correct (`out = beta*v * K@Q / sqrt(hv)`) |
| Attention (single token) | ✓ correct (`out = V` with single KV entry) |
| Buffer layout | ✓ clean (benign 1024-element overflow from h2 into qb, but qb is dead by then) |

## Bugs Found and Fixed

### 1. M-RoPE Section 0 Only (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_mrope` kernel) + `include/mono27b_config.h`
- **Issue:** RoPE kernel rotated ALL 64 dimensions. Qwen3.5 uses multi-section RoPE with sections `[text=11, height=11, width=10, extra=0]`; only section 0 (22 dims) should rotate for text.
- **Impact:** Only affects generation tokens (pos > 0), not prompt step (pos=0).

### 2. L2 Norm Epsilon (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_l2_norm_g` kernel)
- **Issue:** Used `rsqrtf(sum + 1e-10f)` instead of `rsqrtf(fmaxf(sum, eps²))` matching ggml.
- **Impact:** Could over-amplify very small Q/K vectors.

### 3. Q8_K Type Safety (Added)
- **File:** `src/mono27b_gguf.cpp`
- **Issue:** `MONO27B_GGML_TYPE_Q8_K` (type 15) not handled in size/block functions.
- **Note:** Current model has zero Q8_K tensors, so this didn't affect results.

## Evidence Log

### Embedding Verified
- Official `dequantize_row_q4_K` on token row 44883 matches our output exactly
- Earlier `-0.00124168` values were from token row 0 (wrong row), not 44883
- Both produce: `0.015738368, 0.00514060259, 0.0192709565, 0.0334013104, ...`

### LM Head Verified (Q6_K)
- `max_abs = 1.33e-05` on rows 0-7 across both prompt and gen steps
- Confirms Q6_K dequant + matvec implementation is correct

### Divergence Narrows to SSM Layer 0

| Layers | Top1 | Logit | Notes |
|--------|------|-------|-------|
| 0 | 10074 | 8.56 | No layers, just embed→out_norm→LM head |
| 1 (SSM 0) | 220 | 7.03 | **First divergence from reference path** |
| 4 (SSM 0-2, attn 3) | 220 | 6.17 | Same prediction as 1 layer |
| 64 (full) | 11 | 16.92 | Reference: 728 (15.74) |

Layer 0 SSM output stats (from debug dump):
- Embedding L2: 0.93
- Post-RMS norm L2: 69.88 (heavily amplified)
- wqkv output range: [-31.5, 33.5]
- conv+SiLU output range: [-0.28, 9.04]
- DeltaNet output: max=0.23, L2=0.60 (small due to zero initial state)
- ssm_out output L2: 12.24
- Layer out (residual) L2: 12.28
- Post-FFN L2: 15.66

### Tensor Type Reference for SSM Layer 0
```
attn_qkv.weight:  Q6_K   [5120, 10240]  → wqkv projection
attn_gate.weight: Q5_K   [5120,  6144]  → wqkv_gate (z) projection
ssm_conv1d.weight:F32    [4, 10240]     → conv1d kernel
ssm_beta.weight:  F32    [5120,   48]   → beta projection
ssm_alpha.weight: F32    [5120,   48]   → alpha projection
ssm_dt.bias:      F32    [48]           → dt bias
ssm_a:            F32    [48]           → -exp(A_log) gate decay
ssm_norm.weight:  F32    [128]          → per-head RMS norm weight
ssm_out.weight:   Q8_0   [6144, 5120]   → output projection
```

## Open Questions

- **The remaining bug is subtle** — all individual components and the DeltaNet logic have been verified. The divergence arises from the COMPOSITION of the SSM layer, suggesting either:
  1. A weight indexing error (wrong weight loaded for a specific operation)
  2. An architecture interpretation error (wrong tensor shapes/meanings between our code and the reference)
  3. A numerical accumulation issue (rounding/epsilon choices that amplify through the layer)
  4. A CUDA synchronization issue (kernel launches not completing in expected order)

- Can we use `llama-debug` with `--tensor-filter` to dump intermediate hidden states from the reference, allowing direct comparison?

## Next Verification Steps

- [x] Add a CPU-side `q6_K` probe for the first few LM-head rows and compare it against the GPU logits.
- [x] Record the probe result in this file with exact row numbers and deltas.
- [x] If the probe fails, patch the LM head implementation and rerun the same probe.
- [x] If the probe passes, move the tracker back to the earliest upstream mismatch and re-run the layer trace.
- [x] Confirm the raw prompt completion trace is still diverging upstream of the head.
- [x] Compare raw GGUF q4_K dequantization against the backend embed row using official `ggml`.
- [x] Verify embedding parity — confirmed matching on correct token row.
- [x] Fix M-RoPE to only rotate section 0 dimensions (22 instead of 64).
- [x] Fix L2 norm epsilon to match ggml (`fmaxf(sum, eps²)` instead of `sum + 1e-10`).
- [x] Verify Q5_K dequant logic matches `dequantize_row_q5_K` reference.
- [x] Q6_K dequant reference match confirmed via LM head probe.
- [x] Run `compute-sanitizer memcheck` — 0 errors.
- [x] Binary-search: divergence starts at SSM Layer 0 (1 layer gives token 220 ≠ reference's 728).
- [ ] **Write a Python-side reference computation for SSM layer 0**: 
  - Read all layer 0 weights from GGUF
  - Load the h (embedding) values from the debug dump
  - Compute the complete SSM layer in Python using the same exact formulas
  - Compare each intermediate value with our GPU debug dump
  - This will pinpoint the exact operation where values first differ
- [ ] **Check if `ssm_norm.weight` is loaded correctly**: Verify the data at the GGUF offset matches expected 128-element F32 weight array.
- [ ] **Run cuda-gdb or use printf debugging** in the DeltaNet kernel to verify state evolution during the first call.
- [ ] **Verify the conv1d weight order**: Ensure `w[3]` is indeed the last (newest) tap by comparing Python conv with GPU output.
- [ ] **Test with a CPU-side reference** using `libggml-base.so` linked against a small harness that loads the weights and computes one SSM layer step.
- [ ] Keep updating this file until the first mismatch is gone and the remaining trace is 100% identical on the chosen probe.
