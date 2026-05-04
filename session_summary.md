# Session Summary: mono27b / llama.cpp parity debugging

## Goal
Bring `mono27b` into trace-level parity with `llama.cpp` for the same GGUF model, prompt, seed, context, and sampler settings.

## Current State
- We are debugging the raw completion probe, not the chat wrapper path.
- The prompt string is the visible raw prompt `"give"`.
- Both sides tokenize that prompt to the same token id: `44883`.
- `llama.cpp` completion trace for the probe still predicts prompt top1 token `728` with best logit `15.7379`.
- Our backend still predicts prompt top1 token `11`, so the divergence remains real.
- The LM-head probe is no longer the leading suspect.
- The remaining mismatch is upstream of the head, and the strongest evidence points to q4_K tensor handling or runtime use.

## What We Verified
### Prompt and reference trace
- Confirmed the raw prompt tokenization in `mono27b_chat` matches the reference completion trace.
- Confirmed that the reference comparator for this probe is `llama-completion`, not `llama-debug`.
- Found that `llama-debug` can show a misleading `model.input_embed` row for this case because it is not the same trace path as the completion probe.

### q4_K tensor investigation
- Verified the GGUF metadata for `token_embd.weight`.
- `token_embd.weight` is `q4_K`.
- Tensor shape is `5120 x 248320`.
- Tensor size is `715161600`.
- Tensor offset is `1042964480`.
- GGUF data offset is `10993888`.
- Built a tiny standalone C program linked against `ref/llama.cpp/build/bin/libggml-base.so`.
- Used the official `dequantize_row_q4_K` helper on the raw GGUF bytes for token row `44883`.
- Official `ggml` dequant result for token `44883` begins with:
  - `-0.00124168396 -0.0216443539 0.0191609859 -0.0158150196 0.0133316517 0.00750231743 0.0045876503 -0.00707101822`
- Compared neighboring token rows as a sanity check:
  - token `44882` begins with `0.0100064278 -0.00143671036 0.00714564323 0.0042848587 0.0128672123 0.0100064278 -0.00429749489 0.00142407417`
  - token `44884` begins with `0.00984811783 0.00223779678 -0.02312994 -0.00283575058 0.00984811783 -0.000298976898 -0.00537252426 0.0123848915`
- Our backend embed dump for the same prompt row still starts with:
  - `0.015738368,0.00514060259,0.0192709565,0.0334013104,-0.0195875168,0.00867319107,0.015738368,-0.00898975134,...`
- That means the raw row used by our backend does not match the raw row interpreted by the official `ggml` runtime.

### Prompt-prefix experiments
- Tried adding a BOS/EOT-style prefix to `src/mono27b_chat.cpp`.
- Tried both `eos_id` and `bos_id`.
- Reverted both experiments after confirming they did not move the completion trace toward parity.
- The visible prompt still tokenizes as `44883`, so the prefix idea was not the fix.

### q4_K host/runtime experiments
- Added a host-side q4_K embedding helper in `src/mono27b_executor.cu`.
- Added an exact fp16 conversion helper and used it on the q4_K embed host path.
- These changes compiled, but they did not change the observed raw prompt output enough to fix parity.
- Also added a typed F16 matvec path and a typed F16 conv1d path earlier in the investigation; those were useful cleanup, but they did not resolve this current mismatch.

## Debugging Process
### 1. Start from the raw probe
- Used the simplest reproducible probe possible:
  - prompt: `"give"`
  - seed: `944990222`
  - context: `4096`
  - generation: `1`
- This kept the search space small and made trace diffs easier to read.

### 2. Compare completion traces, not just final text
- Ran `mono27b_chat` with trace and debug TSV output.
- Ran `llama-completion` with trace output on the same model and prompt.
- Compared the traces instead of relying on emitted text.
- This showed that the prompt tokens matched, but the logits still diverged.

### 3. Rule out chat-wrapper confusion
- The earlier session history had stale conclusions about chat formatting parity.
- For this probe, the real comparator is the raw completion path.
- Once that was clear, we stopped treating chat-template alignment as the blocker.

### 4. Check the LM head independently
- We previously added a CPU-side Q6_K probe for the first LM-head rows.
- That probe showed the LM head itself was numerically aligned on the probe rows.
- Result: do not keep patching the head when the divergence is upstream.

### 5. Inspect q4_K directly with official ggml code
- To remove ambiguity, we dequantized the raw file bytes with `ggml` itself.
- That proved the file row values as interpreted by official `ggml` do not match what our backend is currently producing for the raw prompt path.
- This is the strongest evidence we have for the remaining bug location.

### 6. Try and discard speculative prompt fixes
- We tested BOS/EOT prefixing.
- We reverted it after it failed to improve parity.
- This prevented the investigation from drifting into prompt-metadata tweaks that were not solving the trace mismatch.

## Commands Run
### Build and execution
```bash
rtk make build
```

```bash
rtk ./build/mono27b_chat \
  -m /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -p "give" \
  --gen 1 \
  --ctx 4096 \
  --seed 944990222 \
  --trace /tmp/mono_give_q4kfix3.trace.tsv \
  --debug /tmp/mono_give_q4kfix3.debug.tsv
```

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

### Official q4_K dequant check
- Built a temporary C helper linked against `ref/llama.cpp/build/bin/libggml-base.so`.
- Called `dequantize_row_q4_K` directly on the raw GGUF bytes for token row `44883`.

## Files Touched in This Investigation
- `src/mono27b_chat.cpp`
  - Experimental BOS/EOT prefix logic was added and then reverted.
- `src/mono27b_executor.cu`
  - Added a host-side q4_K embedding helper.
  - Added an exact fp16 conversion helper.
  - Earlier in the investigation, added F16 matvec and F16 conv1d support.
- `docs/superpowers/plans/2026-05-04-mono27b-llama-parity.md`
  - Contains older tracker notes that are now stale relative to the q4_K findings.

## What We Know Now
- Prompt tokenization is correct.
- The raw completion probe still diverges from `llama.cpp`.
- The LM head is not the current blocker.
- The mismatch is in q4_K tensor handling, dequantization, or how the runtime uses the loaded q4_K weights.
- A larger fix is likely needed than another prompt or sampler tweak.

## Remaining Blocker
- Align our q4_K load/dequant/runtime path with `ggml`'s runtime representation or repacking behavior.
- Do not keep tweaking prompt text or LM-head math unless a new trace proves they are the first divergence again.

## Verification Status
- `rtk make build` passes.
- The raw `"give"` probe still does not match the reference trace.
- The official `ggml` q4_K dequant comparison confirms the backend mismatch is real and upstream of the head.
