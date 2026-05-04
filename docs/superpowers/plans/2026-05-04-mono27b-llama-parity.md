# mono27b / llama.cpp Parity Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach step-by-step parity between `mono27b` and `llama.cpp` on the same model, prompt, seed, context, and sampler settings.

**Architecture:** Compare the pipeline in order: prompt rendering/tokenization, embedding, each transformer/SSM block, final `output_norm`, LM head logits, and sampler output. The tracker records the first diverging layer or step and gets updated after every targeted run so the search space shrinks monotonically.

**Tech Stack:** C++/CUDA, `rtk`, `mono27b_chat`, `llama-completion`, TSV debug traces, GGUF metadata.

---

## Current State

- Same GGUF model file loads successfully on both sides.
- Prompt tokenization matches on the raw-completion probe (`"give"` → token `44883`).
- Hidden state matches through `output_norm` on the raw-completion probe.
- Remaining blocker is the `q6_K` LM head path that produces divergent logits/top token.
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
| Attention / SSM blocks | match through `output_norm` | match | Current evidence says the drift is after the stack |
| Final `output_norm` | match | match | Values agree on the raw probe |
| LM head (`output.weight`) | unresolved | match | Current suspect: `q6_K` matvec or row layout |
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
- Mono debug on raw-completion probe shows the same `output_norm` vector prefix as the reference dump.

## Open Questions

- Is the `q6_K` LM head math wrong, or is the row/stride selection wrong?
- Are we comparing identical prompt forms everywhere, or did one side still include extra wrapper tokens?
- If the LM head matches a CPU re-evaluation of the same weights, the remaining issue is upstream and the tracker must move back one step.

## Next Verification Steps

- [ ] Add a CPU-side `q6_K` probe for the first few LM-head rows and compare it against the GPU logits.
- [ ] Record the probe result in this file with exact row numbers and deltas.
- [ ] If the probe fails, patch the LM head implementation and rerun the same probe.
- [ ] If the probe passes, move the tracker back to the earliest upstream mismatch and re-run the layer trace.
- [ ] Keep updating this file until the first mismatch is gone and the remaining trace is 100% identical on the chosen probe.

