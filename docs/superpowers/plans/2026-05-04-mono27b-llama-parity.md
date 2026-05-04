# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata, `llama-debug` with `--tensor-filter`.

---

## Current State (as of 2026-05-04 Session 3)

**Output still diverges:**
- Reference `llama-debug`: prompt top1 token `728`, logit `15.74`
- Our backend: prompt top1 token `11`, logit `16.92`
- **Divergence starts at SSM Layer 0** (1 layer → token 220 vs 0 layers → token 10074)

### Critical Finding: Output Scale Mismatch

Reference layer 0 outputs are **~10-270x smaller** than ours:

| Tensor | Ref L2 | Our L2 | Ratio (Our/Ref) | Description |
|--------|--------|--------|----------|-------------|
| `attn_output`/`deltanet` | 0.234 | 0.595 | 2.5x | DeltaNet output |
| `final_output`/`rms_gated` | 0.077 | 4.57 | 59x | After RMS norm * gate |
| `linear_attn_out`/`ssm_out` | 0.045 | 12.24 | 272x | After ssm_out projection |
| `l_out`/`post_ffn` | 0.17 | 15.66 | 92x | Layer output after FFN |

The RMS norm amplification is correct (verified 3e-7). The Q6_K and Q5_K matvecs are verified (1e-6 to 4e-7). The divergence is in the COMPOSITION — how the SSM layer chains operations together.

### All Components Verified Numerically

| Component | Status | Method |
|-----------|--------|--------|
| Embedding (Q4_K dequant) | ✓ match | `dequantize_row_q4_K` on GGUF data |
| RMS norm | ✓ 3e-7 | Full 5120-element CPU vs GPU |
| Q6_K matvec (wqkv) | ✓ 1.4e-6 | GPU vs CPU re-dequant inline probe |
| Q5_K matvec (wqkv_gate) | ✓ 4e-7 | Python GGUF data vs GPU row 0 |
| Q6_K matvec (LM head) | ✓ 1.3e-5 | GPU vs CPU probe on 8 rows |
| Q8_0 matvec | ✓ | Formula matches `dequantize_row_q8_0` |
| Q4_K dequant | ✓ | Matches `dequantize_row_q4_K` |
| SiLU | ✓ | Verified numerically |
| Softplus | ✓ | Formula match |
| Conv1d (1st step) | ✓ | Same last-tap weight usage |
| L2 norm | ✓ | Epsilon fix applied |
| DeltaNet logic | ✓ | Matches reference gated_delta_net_cuda |
| M-RoPE | ✓ | Only section 0 (22 dims) |
| compute-sanitizer | ✓ | 0 memory errors |

## Bugs Fixed

1. **M-RoPE Section 0** — Only rotate 22 dims instead of 64
2. **L2 Norm Epsilon** — Changed from `sum + 1e-10` to `fmaxf(sum, eps²)`
3. **Q8_K Type Safety** — Added handling for type 15

## Verification Scripts (`debug/verify/`)

| Script | Purpose |
|--------|---------|
| `test_q5k_matvec.py` | Q5_K matvec verification (GPU vs GGUF) |
| `verify_q6k_wqkv.py` | Q6_K matvec verification (inline GPU vs CPU) |
| `verify_rms_norm.py` | RMS norm verification (full 5120 elements) |
| `verify_deltanet.py` | Python reference DeltaNet implementation |
| `verify_q6k_ref.py` | Q6_K via ggml reference (ctypes stub) |
| `compare_ref.py` | Compare GPU debug vs reference llama-debug dump |
| `ssm_ref_test.cpp` | C++ ggml reference test stub |

## Divergence Data

| Layers | Top1 | Logit | Notes |
|--------|------|-------|-------|
| 0 | 10074 | 8.56 | Embed → output_norm → LM head |
| 1 (SSM 0) | 220 | 7.03 | First divergence |
| 4 (SSM 0-2, attn 3) | 220 | 6.17 | Same top1 |
| 64 (full) | 11 | 16.92 | Ref: 728 (15.74) |

## Next Steps

- [ ] **Parse reference llama-debug output properly** to extract ALL intermediate tensors for layer 0 (qkv_mixed, conv_output_silu, q_conv, attn_output, final_output, linear_attn_out, l_out) and compare element-by-element with our GPU debug dump
- [ ] **Run llama-debug with additional filters** to capture intermediate SSM tensors:
  ```bash
  --tensor-filter "qkv_mixed_transposed|conv_output_silu|q_conv_predelta|attn_output|final_output|linear_attn_out|l_out"
  ```
- [ ] **Fix the attention layer gate** — verify `attn_gate.weight` for attention layers (3,7,11,...) is loaded with shape [5120, 12288] (Q4_K/IQ4_XS), not [5120, 6144] (Q5_K)
- [ ] **Build complete reference SSM layer 0** using `libggml-base.so` on CPU, comparing every intermediate value
- [ ] Keep updating until 100% trace parity
