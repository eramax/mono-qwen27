# Session Summary: mono27b Debugging and Alignment

## Overview
The primary goal of the recent sessions was to align the `mono27b` implementation's prompt construction and generation path with the `llama.cpp` reference implementation to ensure identical output for the same prompt and seed.

## Debugging Process

### 1. Initial Divergence Analysis
- **Observation**: Outputs differed from the reference.
- **Hypothesis**: Prompt/template drift or sampler mismatch.
- **Action**: Compared prompt traces between `mono27b` and `llama.cpp`.
- **Finding**: Found that `mono27b` used a hardcoded chat wrapper and a custom sampler, whereas the reference used GGUF chat-template logic and `common_sampler`.

### 2. Prompt Path Alignment
- **Iteration 1**: Attempted to align prompt strings. Discovered that the model was still drifting into punctuation while the reference produced a thinking preamble.
- **Iteration 2**: Identified that the prompt construction in `mono27b` was not producing the exact same token IDs as the reference.
- **Deep Dive**: Added per-segment token dumps to identify where the mismatch entered (system, user, or assistant segments).
- **Root Cause**: The manual token assembly was not correctly handling special tokens like `<|im_start|>` and `<|im_end|>`.

### 3. Sampler and State Verification
- **Observation**: Even after prompt alignment, the first generated token diverged.
- **Hypothesis**: Sampler mismatch or state handling bug in `mono27b_engine_decode_step`.
- **Finding**: Logits matched through the prompt, but the first decode step diverged, pointing to a computation path issue rather than just sampling parameters.

### 4. Final Fix and Verification
- **Implementation**: Replaced hardcoded chat strings with manual prompt-token assembly in `src/mono27b_chat.cpp`.
- **Changes**:
    - Used explicit `<|im_start|>`, `<|im_end|>`, and newline token boundaries.
    - Trimmed trailing whitespace from user prompts.
- **Verification**:
    - Ran `rtk diff` on trace files (`/tmp/mono_verify.trace.tsv` vs `/tmp/llama_sys.trace.tsv`).
    - Result: `[ok] Files are identical`.
    - Confirmed that the rendered prompt, prompt logits, and the first generation step now perfectly match the reference.

## Progress
- [x] Prompt construction aligned with `llama.cpp`.
- [x] Special token boundaries correctly implemented.
- [x] Prompt logits and first generation token verified for exact parity with reference.
- [x] Temporary debug logging cleaned up.
- [ ] Full-sequence identity for long generations (80+ tokens) not yet fully verified against a stable reference run in this environment, though the path is now correct.

## Last Compact Summary
The prompt path is now fixed to match `llama.cpp` exactly.
- **Modified**: `src/mono27b_chat.cpp` (manual prompt-token assembly).
- **Verified**: Exact trace parity for the prompt and first generation step.
- **Result**: The model now produces identical initial outputs to the reference implementation.
