# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State

- Same GGUF model file loads successfully on both sides.
- Prompt tokenization matches on the raw-completion probe (`"give"` → token `44883`).
- `llama.cpp` completion trace still predicts prompt top1 token `728` with best logit `15.7379`.
- Our backend still predicts prompt top1 token `11` (logit `16.9160`), still diverging.
- **Embedding is now confirmed correct** — official `ggml` q4_K dequant of token row `44883` matches our backend output (both start with `0.015738368`). The earlier `-0.00124168` values were from token row 0 (wrong row), not 44883. This closes the embed-side suspicion.
- The divergence is now **solidly in the hidden layers** (SSM blocks or attention blocks).
- With 0 layers (embedding + output_norm + LM head only), top1 prediction is token `10074` (logit `8.56`).
- With 4 layers (SSM 0,1,2 + attention 3), top1 prediction is token `220` (logit `6.17`).
- With full 64 layers, top1 prediction is token `11` (logit `16.92`).
- The divergence accumulates through the layers and compounds with depth.

## Run Contract

Always compare these together:

- Same model file.
- Same raw prompt string.
- Same `ctx`.
- Same seed.
- Same trace/debug format.
- Same tokenization mode.

Preferred probe:

```bash
rtk ./build/mono27b_chat \
  -m /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -p "give" \
  --gen 1 \
  --ctx 4096 \
  --seed 944990222 \
  --trace /tmp/mono_give.trace.tsv \
  --debug /tmp/mono_give.debug.tsv
```

Reference probe:

```bash
rtk ./ref/llama.cpp/build/bin/llama-completion \
  -m /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -p "give" \
  -n 1 \
  -c 4096 \
  -b 1 \
  -ub 1 \
  -no-cnv \
  --trace-file /tmp/ref_give_b1.trace.tsv \
  -lv 4
```

## Bugs Found and Fixed in This Session

### 1. M-RoPE Section 0 Only (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_mrope` kernel) + `include/mono27b_config.h`
- **Issue:** The RoPE kernel rotated ALL 64 M-RoPE dimensions with the token position. Qwen3.5 uses multi-section RoPE with sections `[text=11, height=11, width=10, extra=0]` (total 32 pairs = 64 dims). For text-only inference, only section 0 (11 pairs = 22 dims) should be rotated; sections 1 and 2 use position 0 (no rotation).
- **Fix:** Added `MONO27B_N_ROT_DIMS_S0 = 22` constant and changed `k_mrope` to only process the first 22 dimensions.
- **Impact:** Does NOT affect position-0 tokens (cos=1, sin=0 when pos=0), but WILL affect generation tokens (pos > 0).

### 2. L2 Norm Epsilon Match (Fixed)
- **File:** `src/mono27b_executor.cu` (`k_l2_norm_g` kernel)
- **Issue:** Our kernel used `rsqrtf(sum + 1e-10f)` while ggml uses `rsqrtf(fmaxf(sum, eps²))` where `eps = 1e-6` (RMS epsilon), giving `eps² = 1e-12`.
- **Fix:** Changed to `fmaxf(sum, MONO27B_RMS_EPS * MONO27B_RMS_EPS)` to match ggml.
- **Impact:** Prevents over-amplification of very small Q/K vectors (sum < 1e-12 vs sum < 1e-10).

### 3. GGUF Q8_K Type Handling (Added as safety, not needed for this model)
- **File:** `src/mono27b_gguf.cpp`
- **Issue:** `MONO27B_GGML_TYPE_Q8_K` (type 15) was defined in the enum but not handled in `quant_type_size()` or `quant_block_size()`, causing those tensors to have `size_bytes = 0` and be skipped.
- **Note:** The current model does NOT use Q8_K tensors (confirmed: 0 Q8_K tensors in the file), so this didn't affect this run. Kept as a safety fix for future models.

## Layer Status

| Layer / Step | mono27b status | llama.cpp status | Notes |
| --- | --- | --- | --- |
| GGUF load | match | match | Same file, same tensor metadata |
| Tokenization | match | match | Raw prompt `"give"` maps to token `44883` |
| Prompt path | match | match | Raw-completion probe is aligned |
| Embed | **match** | match | **Resolved.** Official ggml q4_K dequant on correct token row (44883) matches our output. Both produce `0.015738368, 0.00514060259, ...` |
| SSM Layer 0 | unresolved | (ref has no per-layer trace) | First SSM layer produces non-trivial delta net output. RMS norm, conv1d, SiLU verified numerically |
| SSM Layer 1 | unresolved | (ref has no per-layer trace) | |
| SSM Layer 2 | unresolved | (ref has no per-layer trace) | |
| Attention Layer 3 | unresolved | (ref has no per-layer trace) | Verified KV-cache, softmax attention, gate apply all execute. V-only output for first token is correct (single-token attention degenerates to identity). |
| SSM Layer 4+ | unresolved | (ref has no per-layer trace) | Layers 4-63 process, all output finite |
| Final `output_norm` | **match** | match | RMS norm with output_norm weights verified numerically (weights read from GGUF, values around 1.7-2.0) |
| LM head (`output.weight`) | **match** | match | CPU probe matches GPU within `1.3e-5` on rows 0-7. Verified 2-step generation logits also match within `1.3e-5`. |
| Sampler | unresolved | match | Not the leading suspect when LM head is proven |

## Evidence Log

### Embedding Resolution
- Built standalone `check_gguf.cpp` tool linked against `ref/llama.cpp/build/bin/libggml-base.so`.
- Official `dequantize_row_q4_K` on raw GGUF bytes for token row 44883 produces:
  - `0.015738368, 0.00514060259, 0.0192709565, 0.0334013104, ...`
- Our backend embed dump produces IDENTICAL values.
- The earlier `-0.00124168` values were from row 0 (not row 44883) — the `check_gguf` tool was accidentally reading the wrong offset.
- Neighbor rows also confirmed correct:
  - row 44882: `0.0100064278 ...`
  - row 44884: `0.00984811783 ...`

### LM Head Probe (Q6_K)
- Verified at both prompt step and gen step.
- `max_abs = 1.33e-05` on rows 0-7 across both steps.
- Confirms Q6_K dequant + matvec produce correct logits.

### All Tensor Types in GGUF
```
F32 (type 0):    449 tensors
Q8_0 (type 8):    48 tensors
Q4_K (type 12):  207 tensors
Q5_K (type 13):   70 tensors
Q6_K (type 14):   65 tensors
IQ4_XS (type 23): 12 tensors
```
All types are supported by the code. No Q8_K (type 15) tensors exist in this model.

### Verified Individual Components (against ggml/C reference)
| Component | Status | Method |
| --- | --- | --- |
| Q4_K dequant | ✓ match | `dequantize_row_q4_K` on raw GGUF bytes |
| Q6_K dequant + matvec | ✓ match | GPU vs CPU probe on LM head |
| RMS norm | ✓ match | Numerical check from debug values, formula matches ggml kernel |
| SiLU activation | ✓ match | Verified from debug: `silu(0.079052) = 0.0410875` matches `x/(1+exp(-x))` |
| L2 normalization | ✓ match (after fix) | Epsilon changed from `1e-10` to `fmaxf(sum, eps²)` to match ggml |
| M-RoPE | ✓ match (after fix) | Only rotate section 0 (22 dims) for text-only inference |
| Attention (single token) | ✓ correct | V-only output is correct for single-token degenerate case |
| Conv1d (first step) | ✓ correct | `conv_out = inp * w[kernel_size-1]` for zero initial state |
| Q5_K dequant logic | ✓ match | Manual verification of index mapping against `dequantize_row_q5_K` |
| Buffer memory layout | ✓ clean | No overlaps between `h`, `h2`, `qb`, `kb`, `fb`, `sb`, `gb`, `logits` |

### Divergence Experiment: Layer Count
| Layers | Top1 Token | Top1 Logit | Reference (64 layers) |
| --- | --- | --- | --- |
| 0 (no layers) | 10074 | 8.56 | — |
| 4 (SSM 0-2 + attn 3) | 220 | 6.17 | — |
| 64 (full) | 11 | 16.92 | 728 (15.74) |

The divergence grows with layer count but even the first SSM layer already pushes the hidden state in the wrong direction.

## Open Questions

- Which quantized matvec kernel first produces incorrect output? Need to compare Q4_K matvec (FFN gate/up, attention Q/K), Q5_K matvec (SSM gate), Q6_K matvec (FFN down, attention Q for non-first?), Q8_0 matvec (SSM output) against ggml equivalents.
- Is there a subtle CUDA memory issue (e.g., incorrect `__half` alignment in struct, race condition between async kernels)?
- Could the conv state caching be incorrect for subsequent SSM layers within the same prompt step? (All layers share the same pos=0, so conv states are populated per-layer and consumed only on the next token.)
- Is the `ssm_state` (DeltaNet state) initialized correctly to zeros?

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
- [ ] **Binary-search the diverging layer**: Run with increasing layer count (1, 2, 3, 4, 8, 16, 32, 64) and record top1 token + logit. Find the first layer where the prediction diverges from the reference.
- [ ] **Verify quantized matvec kernels**: Write a standalone CUDA test that runs each quantized matvec type (Q4_K, Q5_K, Q6_K, Q8_0) with known inputs and compares against ggml's CPU output from the same raw GGUF data.
- [ ] **Test conv1d kernel isolation**: Replace conv1d with a simple copy (bypass convolution) and see if the output changes toward the reference.
- [ ] **Test DeltaNet isolation**: Replace the DeltaNet SSM block with a simple identity (just copy v to output) and compare.
- [ ] **Check shared memory bank conflicts** in `k_q4k_mv`, `k_q5k_mv`, `k_q6k_mt` kernels that could cause silent numerical corruption.
- [ ] **Add `cuda-memcheck` run** to detect out-of-bounds memory access in any kernel.
- [ ] Keep updating this file until the first mismatch is gone and the remaining trace is 100% identical on the chosen probe.
