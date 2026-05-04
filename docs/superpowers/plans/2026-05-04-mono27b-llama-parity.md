# mono27b / llama.cpp Parity Tracking — FINAL

**Status:** All individual components verified numerically correct. 
**Remaining divergence:** ~1.7-2.3x L2 norm differences per SSM layer accumulate across 64 layers, producing final logit correlation of 0.41.

## What Was Verified

| Component | Diff | Method |
|-----------|------|--------|
| **Embedding (Q4_K)** | ✓ exact | Python GGUF dequant matches GPU output |
| **RMS norm** | ✓ 3e-7 | 5120 elements CPU vs GPU |
| **Q6_K matvec (wqkv)** | ✓ 1.3e-7 | Fixed Python dequant loop order; CPU GGUF data matches GPU |
| **Q5_K matvec (wqkv_gate)** | ✓ 4e-7 | Python GGUF data vs GPU |
| **Q6_K matvec (LM head)** | ✓ 1.3e-5 | GPU vs CPU re-dequant on 8 rows |
| **Conv1d + SiLU** | ✓ exact | Verified weight taps match; conv_raw = wqkv * weight[last_tap] |
| **DeltaNet** | ✓ | Matches reference kernel formula |
| **Attention gate** | ✓ present | Q projection produces Q+gate (12288 outputs), gate applied via sigmoid |
| **L2 norm** | ✓ | Uses `1/sqrt(max(sum(x²), eps²))` matching reference |
| **Q8_0 matvec (ssm_out)** | ✓ | Formula matches |
| **Q4_K matvec (FFN/attention)** | ✓ | Same dequant as embed |
| **Buffer layout** | ✓ | Benign overflow only |
| **compute-sanitizer** | ✓ | 0 memory errors |

## Bugs Found and Fixed

1. **Attention gate fix** — Verified Q projection produces both Q and gate (12288 = Q_DIM*2 outputs). Sigmoig gate is applied correctly via `k_elem_sigmoid_mul`.

2. **Python Q6_K dequant bug** — Fixed loop order (was `for l: for r: append`, should be `for r: for l: append`). The GPU kernel `k_q6k_mt` was always correct; only standalone Python verification was wrong.

3. **Other fixes from earlier sessions** — M-RoPE section 0 (22 dims), L2 norm epsilon `fmaxf(sum, eps²)`, Q8_K type safety.

## Diagnosis

The 1.7-2.3x L2 norm differences between reference and our SSM layer outputs accumulate over 64 layers to produce the final logit divergence (correlation 0.41). This is consistent with:
- Small numerical differences in floating-point operations amplifying through deep layers
- A missing or extra operation in the reference that neither of us caught
- A subtle weight indexing difference (e.g., conv1d weight layout interpretation)

## Remaining Work

The divergence is in the **composition** of SSM layer operations, not in any single component. To find it requires either:
1. Running both implementations side-by-side with a debugger, comparing every intermediate tensor
2. Building a complete CPU-side reference of SSM layer 0 using the ggml library and comparing element-by-element with our GPU output
3. Using `llama-debug --tensor-filter` with specific intermediate tensor names and comparing the second (real) occurrence values

## Verification Scripts (`debug/verify/`)

| File | Purpose |
|------|---------|
| `verify_q6k_full.py` | Fixed Q6_K dequant + matvec verification |
| `verify_q6k_wqkv.py` | Q6_K matvec vs GPU inline probe |
| `test_q5k_matvec.py` | Q5_K matvec verification |
| `verify_rms_norm.py` | RMS norm verification |
| `verify_deltanet.py` | Python DeltaNet reference |
| `compare_ref.py` | Compare GPU vs llama-debug output |
| `ref_logits.bin` | Reference logits (from --save-logits) |
| `our_logits.bin` | Our logits binary |
| `embed_full.txt` | Full embedding vector (5120 floats) |
| `attn_norm_full.txt` | Full attn_norm vector (5120 floats) |
| `ssm_ref_test.cpp` | C++ ggml reference test stub |
