# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State

- Same GGUF model file loads successfully on both sides.
- Prompt tokenization matches on the raw-completion probe (`"give"` → token `44883`).
- `llama.cpp` completion trace for the probe still predicts prompt top1 token `728` with best logit `15.7379`.
- Our backend still predicts prompt top1 token `11`, so the raw probe still diverges.
- The earlier layer-3 gated-attention note is stale for the current probe state.
- The LM-head probe is not the leading suspect right now.
- The strongest evidence points to q4_K tensor handling or runtime use on the input-embedding side.
- The current trace set is good enough to keep narrowing; it is not yet a 100% match.

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

Recent confirmation commands:

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

## Layer Status

| Layer / Step | mono27b status | llama.cpp status | Notes |
| --- | --- | --- | --- |
| GGUF load | match | match | Same file, same tensor metadata |
| Tokenization | match | match | Raw prompt `"give"` maps to token `44883` |
| Prompt path | match | match | Raw-completion probe is aligned |
| Embed | unresolved vs reference | match | Official `ggml` q4_K dequant on token row `44883` does not match our backend embed dump |
| Attention / SSM blocks | unresolved vs reference | unresolved vs reference | Re-localize after fixing q4_K handling; earlier layer-3 note is stale |
| Final `output_norm` | unresolved vs reference | unresolved vs reference | Do not touch the head again until embed-side parity is restored |
| LM head (`output.weight`) | match | match | CPU probe matches GPU within `1.3e-5` on rows 0-7 |
| Sampler | unresolved | match | Not the leading suspect until LM head is proven |

## Evidence Log

- `output.weight` metadata from `llama-gguf`:
  - type: `q6_K`
  - `n_elts = 4966400`
  - `size = 1042944000`
  - `n_dims = 2`
  - `ne = (5120, 248320, 1, 1)`
- `output_norm.weight` metadata from `llama-gguf`:
  - type: `f32`
  - `n_elts = 5120`
  - `size = 20480`
- `mono27b` Q6_K LM-head probe on raw-completion `give`:
  - rows 0-7 compare GPU logits against a CPU recomputation from the same loaded blocks
  - worst row: `2`
  - `max_abs = 1.33514404e-05`
  - sample rows:
    - row 0: gpu `8.7132206`, cpu `8.71322918`, delta `8.58306885e-06`
    - row 1: gpu `3.55735159`, cpu `3.55735612`, delta `4.529953e-06`
    - row 2: gpu `5.28695726`, cpu `5.28696012`, delta `2.86102295e-06`
    - row 3: gpu `4.44023085`, cpu `4.44023085`, delta `0`
- Official `ggml` q4_K dequant of token row `44883` begins with:
  - `-0.00124168396 -0.0216443539 0.0191609859 -0.0158150196 0.0133316517 0.00750231743 0.0045876503 -0.00707101822`
- Neighbor sanity checks on token rows:
  - row `44882` begins with `0.0100064278 -0.00143671036 0.00714564323 0.0042848587 0.0128672123 0.0100064278 -0.00429749489 0.00142407417`
  - row `44884` begins with `0.00984811783 0.00223779678 -0.02312994 -0.00283575058 0.00984811783 -0.000298976898 -0.00537252426 0.0123848915`
- Our backend embed dump for the same prompt row still starts with:
  - `0.015738368,0.00514060259,0.0192709565,0.0334013104,-0.0195875168,0.00867319107,0.015738368,-0.00898975134,...`
- That is the current evidence for the remaining mismatch.

## Open Questions

- Which upstream block first diverges now that the Q6_K LM head has been cleared?
- Are we comparing the same q4_K runtime representation on the embedding side, or is one side repacking/dequantizing differently?
- Do we need to mirror `ggml`'s q4_K load path exactly instead of patching individual consumers?
- Can we localize the first divergence again after the embed path is fixed?

## Next Verification Steps

- [x] Add a CPU-side `q6_K` probe for the first few LM-head rows and compare it against the GPU logits.
- [x] Record the probe result in this file with exact row numbers and deltas.
- [x] If the probe fails, patch the LM head implementation and rerun the same probe.
- [x] If the probe passes, move the tracker back to the earliest upstream mismatch and re-run the layer trace.
- [x] Confirm the raw prompt completion trace is still diverging upstream of the head.
- [x] Compare raw GGUF q4_K dequantization against the backend embed row using official `ggml`.
- [ ] Keep updating this file until the first mismatch is gone and the remaining trace is 100% identical on the chosen probe.
