# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State

- Same GGUF model file loads successfully on both sides.
- Prompt tokenization matches on the raw-completion probe (`"give"` → token `44883`).
- Q6_K LM-head math matches an internal CPU re-evaluation on the raw-completion probe.
- The end-to-end logits still diverge from `llama.cpp`, so the remaining mismatch is upstream of the head.
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
  -no-cnv \
  --trace-file /tmp/ref_give.trace.tsv
```

## Layer Status

| Layer / Step | mono27b status | llama.cpp status | Notes |
| --- | --- | --- | --- |
| GGUF load | match | match | Same file, same tensor metadata |
| Tokenization | match | match | Raw prompt `"give"` maps to token `44883` |
| Prompt path | match | match | Raw-completion probe is aligned |
| Embed | match | match | Hidden-state traces line up on the probe |
| Attention / SSM blocks | unresolved vs reference | unresolved vs reference | Needs a fresh layer-by-layer trace now that the LM head is cleared |
| Final `output_norm` | unresolved vs reference | unresolved vs reference | Re-localize the first mismatch before touching the head again |
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
- Legacy notes suggested `output_norm` agreement on an earlier probe; revalidate that claim on the current single-token run before using it as a blocker gate.
- `mono27b` Q6_K LM-head probe on raw-completion `give`:
  - rows 0-7 compare GPU logits against a CPU recomputation from the same loaded blocks
  - worst row: `2`
  - `max_abs = 1.33514404e-05`
  - sample rows:
    - row 0: gpu `8.7132206`, cpu `8.71322918`, delta `8.58306885e-06`
    - row 1: gpu `3.55735159`, cpu `3.55735612`, delta `4.529953e-06`
    - row 2: gpu `5.28695726`, cpu `5.28696012`, delta `2.86102295e-06`
    - row 3: gpu `4.44023085`, cpu `4.44023085`, delta `0`

## Open Questions

- Which upstream block first diverges now that the Q6_K LM head has been cleared?
- Are we comparing identical prompt forms everywhere, or did one side still include extra wrapper tokens?
- Do we need a fresh per-layer trace around the first divergent block before changing any more kernels?

## Next Verification Steps

- [x] Add a CPU-side `q6_K` probe for the first few LM-head rows and compare it against the GPU logits.
- [x] Record the probe result in this file with exact row numbers and deltas.
- [x] If the probe fails, patch the LM head implementation and rerun the same probe.
- [x] If the probe passes, move the tracker back to the earliest upstream mismatch and re-run the layer trace.
- [ ] Keep updating this file until the first mismatch is gone and the remaining trace is 100% identical on the chosen probe.
