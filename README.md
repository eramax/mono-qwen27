# mono27b

Clean-room workspace for the monolithic Qwen3.6-27B CUDA executor.

## Purpose

`mono27b/` is intentionally separate from the root `specfusion` runtime. It keeps the new blob-first CUDA executor isolated from `ggml`-shaped runtime assumptions while the root engine remains the correctness reference.

## Build

```bash
rtk cmake -B build -DCMAKE_BUILD_TYPE=Release
rtk cmake --build build --target mono27b_chat mono27b_pack -j$(nproc)
```

## Smoke Tests

```bash
rtk bash tests/smoke_mono27b_scaffold.sh build
rtk bash tests/smoke_mono27b_pack.sh build/mono27b_pack
rtk bash tests/smoke_mono27b_chat.sh build/mono27b_chat build/mono27b_pack
```

## Current Contract

- target quant: `q4_k_m`
- draft quant: `q8_0`
- runtime input: `.m27b` packed blob
- execution binary: `mono27b_chat`
- offline packer: `mono27b_pack`

## Current Status

What works now:
- local GGUF parsing without `ggml`
- vendored tokenizer encode/decode in the clean-room tree
- blob packing into `.m27b`
- CUDA-backed prompt embedding gather and next-token candidate scoring from `token_embd.weight`
- `mono27b_chat -m <target.gguf> -md <draft.gguf> -p "..."`

What remains:
- full target and draft forward passes
- KV/SSM cache integration and rollback
- higher-quality sampling for general chat prompts
- replacing the current partial decode loop with a faithful model executor
