# mono27b vs llama.cpp — Analysis & Plan

Generated: 2026-05-05
Command: `make e2e-text-all`

## Findings from `make e2e-text-all`

### 1. ❌ Generated Text is Completely Different

| Metric | Value |
|--------|-------|
| Similarity (1-LD/len) | 0.219 |
| Our length | 1898 chars |
| Ref length | 4123 chars |
| First divergence | character 0 — **texts share no common prefix** |

**Our output**: `Here are two examples of .py (Python) code snippets...`
**Ref output**: `<think>\nHere's a thinking process:\n\n1. **Understand User Request:**...`

The reference (llama-completion with `-cnv`) applies the model's Jinja chat template, which adds `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`. Our `--chat` flag does the same with a hardcoded equivalent template (confirmed: last prompt token is `<think>` 248068). **Despite the same prompt prefix, the model generates different first tokens.**

### 2. ⚠️ E2E Logits — Not Identical (Prompt Token Only)

| Metric | Value |
|--------|-------|
| Top1 token | 728 (both match) |
| Correlation | 0.980010 |
| MSE | 0.4326 |

Current E2E comparison only checks the **first prompt token** (`"give"`). The **last prompt token** (where generation starts, after `<think>`) is not compared. A correlation of 0.98 with MSE 0.43 on a 248k-vocab distribution is enough to flip top-1 tokens during sampling.

### 3. ✅ Tensor Comparisons — All PASS

All 20 intermediate tensors match the reference within tolerance at 5e-5 absolute diff. This means individual layer computations are correct. The logit difference likely accumulates from small errors in each layer.

### 4. ❌ Q6_K Matvec — FAIL (N/A)

The Q6_K matrix-vector test fails with "N/A". Need to investigate the test or the underlying kernel.

### 5. ⚠️ Speed: ~17 tok/s vs ~35 tok/s (2× slower than llama.cpp)

From test run:
- Prefill: 16.9 tok/s (947.5ms for 16 tokens)
- Generation: 17.4 tok/s
- GPU: RTX 3090 (same GPU for both)

---

## Investigation Plan

### Priority 1: Logit Debugging — Why does the first generated token differ?

- [ ] **Compare logits at the last prompt position** (after `<think>`), not just the first token. Modify `gen-our` / `gen-ref` to capture logits at the generation position.
- [ ] **Layer-by-layer correlation** at the generation position — find which layer first shows significant divergence.
- [ ] **Check BOS token handling**: Our code adds BOS if `add_bos_token` is true. Verify llama.cpp does the same with `-b 1 -ub 1`.
- [ ] **Check position encoding**: Verify M-RoPE is applied identically at every position.
- [ ] **Verify SSM state initialization**: DeltaNet/S6 state initialization must match llama.cpp exactly.
- [ ] **Check `should_skip_token` behavior**: The decode function skips `<|im_start|>`, `<|im_end|>`, and `<|endoftext|>` — verify this doesn't affect the output text.

### Priority 2: Performance Profiling — Where is the 2× slowdown?

- [ ] **Profile kernel launch times**: Measure CUDA kernel duration vs llama.cpp for attention, SSM, FFN.
- [ ] **Check CUDA graph usage**: llama.cpp may use CUDA graphs for faster kernel launches.
- [ ] **Check batch size**: Our code processes 1 token at a time. Verify llama.cpp's `-b 1 -ub 1` also processes single-token.
- [ ] **Quantization kernel differences**: Compare Q4_K_M matmul kernel implementation.
- [ ] **Check GPU utilization**: Are we saturating the GPU or are there sync points?

### Priority 3: Fix Failing Tests

- [ ] **Fix Q6_K matvec test**: It shows "N/A" — check if the tensor name or path has changed.
- [ ] **Add logit comparison at generation position** to E2E test suite.
- [ ] **Add timing comparison** to `compare-text` or a new performance test target.

### Priority 4: Template Verification

- [ ] **Compare template output** character-by-character between our hardcoded template and llama-completion's Jinja-rendered template.
- [ ] **Verify special token IDs**: Double-check im_start_id, im_end_id, think_id are correct.
- [ ] **Check if `<|im_end|>` should terminate the assistant prefix** — the template puts `<|im_start|>assistant\n<think>\n` without a closing `<|im_end|>`, which is correct (generation is open-ended).

### Priority 5: Shared Context / Attention Mask

- [ ] **Verify causal attention mask** matches llama.cpp exactly (especially for SSM layers with convolution).
- [ ] **Check position IDs** for each token: our M-RoPE test passed, but verify for the full chat template sequence.

---

## Root Causes Identified

### 1. Tokenization differences
Our BPE tokenizer vs llama.cpp's produces DIFFERENT token sequences for the same text.
This means the model receives different input tokens → different positions → different behavior.

Example: "give me 2 py example"
- Ours: `give` `Ġme` `Ġ` `2` `Ġpy` `Ġexample` = **6 tokens**
- Ref: likely **4 tokens** (based on 14 vs 16 total for chat prompt)

### 2. Internal state divergence
Even with same single token "give" (token 44883), logit correlation is only **0.98** (MSE 0.43).
The first 2 greedy tokens match (" me" + " a"), but by step 2 they diverge.
This means the KV cache / SSM state accumulates errors.

### 3. No CUDA graphs
llama.cpp uses `USE_GRAPHS = 1` (CUDA graphs) — gives ~2× speedup by replaying fixed
compute graphs. Our implementation launches kernels individually each step.

### 4. Template mismatch (reverted)
The `<think>\n` prefix was added to match llama.cpp's Jinja template, but the model
doesn't generate thinking content because the core computation diverges first.

## Action Items

### Priority 1: Performance — Add CUDA Graph Support
- Biggest single win for speed (~2×)
- Capture the decode step as a CUDA graph after the first invocation
- Replay the graph for subsequent tokens

### Priority 2: Fix BPE Tokenizer
- Compare our BPE merge output with llama.cpp's for common words
- Fix chunk splitting in encode() to match GPT-2 BPE behavior

### Priority 3: Template Alignment
- Once core computation matches, re-add `<think>\n` prefix
- Or better: implement a simple template that exactly matches llama.cpp's Jinja output

## Notes

- The model is `Qwen3.6-27B-UD-Q4_K_XL.gguf` (Qwen 3.6 with UD architecture, 27B params, Q4_K_XL quantization)
- GPU: NVIDIA GeForce RTX 3090 (24GB)
- Reference build: `ref/llama.cpp/` (latest)
- Our build: `build/mono27b_chat`
