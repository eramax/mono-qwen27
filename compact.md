Goal
- Fix make test in /mnt/mydata/projects2/mono27b so the model generates valid coherent text instead of multilingual garbage.
Constraints & Preferences
- Standalone app, no external dependencies (no ggml as dependency)
- Can borrow functions/code from ggml (exists at /mnt/mydata/projects2/specfusion/deps/ggml/)
- Mega kernel approach preferred (fused operations per layer)
- Reference project: /mnt/mydata/projects2/specfusion
Progress
Done
- GGUF reader: supports all quant types (F32,F16,Q4_K,Q5_K,Q6_K,Q8_0,IQ4_XS=type23) with correct type_size/block_size and tensor dims
- Weight loading: WeightView struct with type-aware dispatch, no memory leaks
- Memory management: state allocated before weights, unified work buffer per step
- CUDA kernels: Q4_K matvec (works), Q5_K matvec (written, untested correctness), Q6_K matvec (strided layout fixed), Q8_0 matvec, IQ4_XS matvec, RMS norm, elementwise ops, M-RoPE
- Attention kernel: KV cache layout fixed from [head][pos][dim] to [pos][head][dim], overflow-safe softmax
- Chat template: Qwen3 <|im_start|>user/assistant wrapping added
- Tensor name lookups: attn_output.weight fallback for attn_o.weight, attn_qkv.weight/attn_gate.weight for SSM, ssm_dt.bias/ssm_a for SSM small weights
- Zero-layer test proves embedding + output_norm + LM head work correctly (produces "ingles")
- Attention identity test proves Q path works, V path corrupts output (V projection via Q6_K matvec suspect)
- DeltaNet state layout bug just fixed: col + i*hv → col*hk + i (transposed storage)
In Progress
- Fixing SSM Conv1D and DeltaNet kernels — remaining sources of garbage
- DeltaNet state layout fix not yet tested
Blocked
- SSM pipeline broken: with Conv1D+DeltaNet enabled → NaN; without them → pass-through produces garbage
- Output quality limited by SSM kernel correctness
Key Decisions
- IQ4_XS (type 23) needed because attention layers 15,51,59,63 use it for attn_q/attn_k weights — added block struct (136 bytes), kvalues_iq4nl table, and matvec kernel
- block_size function renamed to quant_block_size to avoid collision with CUDA header macro
- DeltaNet g_raw clamped to -80,80 to prevent exp overflow (NaN source)
- Attention softmax clamped (maxv ≤ 64, exp arg ≥ -64) for numerical stability
- SSM state allocated eagerly (2.3GB) rather than lazily — survives memory fragmentation
Next Steps
1. Test the just-fixed DeltaNet state layout (col*hk + i indexing)
2. Verify if Q6_K and Q5_K matvec kernels produce correct output (compare with specfusion reference for a single layer)
3. If SSM still broken, study /mnt/mydata/projects2/specfusion/deps/ggml/src/ggml-cuda/ssm-conv.cu and gated_delta_net.cu for reference implementation
4. Compare full pipeline token-by-token with specfusion to identify exact divergence point
Critical Context
- Model: Qwen3.6-27B-UD-Q4_K_XL (hybrid: 16 attn + 48 SSM layers, hidden=5120, vocab=248320)
- Model file: /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf (17GB)
- Zero-layer output "ingles" proves embedding, output_norm, and LM head (Q6_K) are all correct
- With attention → garbage; without attention → English-like ("becausebecausealan Karr Coimbra")
- SSM layer types mixed: wqkv=Q6_K, wqkv_gate=Q5_K, ssm_out=Q8_0, ffn_down=Q6_K(SSM) or Q4_K(attn)
- Output norm weight: output_norm.weight (F32)
- LM head: output.weight (Q6_K, 248320×5120)
Relevant Files
- src/mono27b_executor.cu — all CUDA kernels (matvec, attention, SSM, elementwise) and engine API
- src/mono27b_chat.cpp — main CLI with chat template, GGUF loading, decode loop
- include/mono27b_config.h — model constants, WeightView, executor state structs
- include/mono27b_gguf.h — GGUF reader types and tensor info (added IQ4_XS enum)
- src/mono27b_gguf.cpp — GGUF reader with quant type size/block functions
- include/mono27b_tokenizer.h — tokenizer interface
- /mnt/mydata/projects2/specfusion/deps/ggml/src/ggml-cuda/ssm-conv.cu — reference Conv1D kernel
- /mnt/mydata/projects2/specfusion/deps/ggml/src/ggml-cuda/gated_delta_net.cu — reference DeltaNet kernel