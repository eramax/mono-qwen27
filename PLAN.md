# mono27b Fix Plan

## Problem
The executor skips all 64 transformer layers. Inference is:
```
embed → mean-pool → RMS norm → LM head
```
Instead of:
```
embed → 64×{norm → [attn|SSM] → residual → norm → FFN → residual} → out_norm → LM head
```

## Architecture: Qwen3.5-27B Hybrid
- 64 layers total
- 16 full-attention layers (every 4th: il=3,7,11,...,63)
- 48 Gated DeltaNet SSM layers (all others)
- Hidden dim: 5120
- Head dim: 256, 24 query heads, 4 KV heads (GQA ratio=6)
- M-RoPE: 4 sections {11,11,10,0} pairs = 64 dims rotated
- FFN: SwiGLU, intermediate 17408
- Vocab: 248320

## Approach
1. Load GGUF directly (no external deps, borrow kernels from ggml source)
2. Keep existing Q4_K/Q6_K/Q8_0 matvec kernels for weight projections
3. Fuse each layer ops into a single mega kernel
4. Borrow gated_delta_net + ssm_conv from ggml

## Mega Kernel Structure
```
attention_layer_kernel(hidden, layer_weights, kv_cache, pos)
  └─ rms_norm(hidden, attn_norm) → normed
  └─ Q = wq @ normed, K = wk @ normed, V = wv @ normed
  └─ qk_rms_norm(Q, q_norm), rms_norm(K, k_norm)
  └─ mrope(Q), mrope(K)
  └─ KV cache write(pos, K, V)
  └─ attention(Q, K_cache[0..pos], V_cache[0..pos])
  └─ gate = sigmoid(gate_part_of_Q)
  └─ hidden_out = wo @ (attn * gate) + hidden  (residual)
  └─ rms_norm(hidden_out, post_norm) → ffn_in
  └─ gate = silu(ffn_gate @ ffn_in), up = ffn_up @ ffn_in
  └─ hidden = ffn_down @ (gate * up) + hidden_out  (residual)

ssm_layer_kernel(hidden, layer_weights, conv_state, ssm_state)
  └─ rms_norm(hidden, attn_norm) → normed
  └─ qkv = wqkv @ normed, z = wqkv_gate @ normed
  └─ beta = sigmoid(ssm_beta @ normed)
  └─ alpha = softplus(ssm_alpha @ normed + dt_bias)
  └─ g = alpha * ssm_a_log
  └─ conv1d(qkv, conv_state, ssm_conv1d) → conv_out
  └─ split conv_out → q, k, v
  └─ l2_norm(q), l2_norm(k)
  └─ gated_delta_net(q, k, v, g, beta, state) → ssm_out
  └─ hidden = ssm_out_proj @ (rms_norm(ssm_out) * silu(z)) + hidden
  └─ rms_norm(hidden, post_norm) → ffn_in
  └─ gate = silu(ffn_gate @ ffn_in), up = ffn_up @ ffn_in
  └─ hidden = ffn_down @ (gate * up) + hidden

output_kernel(hidden, output_norm_weight, lm_head_weight) → logits
```

## Weight Tensors per Layer

### Attention Layer (16 total, per layer):
| Tensor | Shape | Type |
|--------|-------|------|
| attn_norm.weight | [5120] | F32 |
| attn_q.weight | [6144, 5120] | Q4_K |
| attn_k.weight | [1024, 5120] | Q4_K |
| attn_v.weight | [1024, 5120] | Q4_K |
| attn_o.weight | [5120, 6144] | Q4_K |
| attn_q_norm.weight | [256] | F32 |
| attn_k_norm.weight | [256] | F32 |
| post_attention_norm.weight | [5120] | F32 |
| ffn_gate.weight | [17408, 5120] | Q4_K |
| ffn_up.weight | [17408, 5120] | Q4_K |
| ffn_down.weight | [5120, 17408] | Q4_K |

### SSM Layer (48 total, per layer):
| Tensor | Shape | Type |
|--------|-------|------|
| attn_norm.weight | [5120] | F32 |
| ssm_in/attn_q.weight | [10240, 5120] | Q4_K |
| ssm_in_gate/attn_k.weight | [6144, 5120] | Q4_K |
| ssm_conv1d.weight | [10240, 4] | F16 |
| ssm_beta.weight | [48, 5120] | F16 |
| ssm_alpha.weight | [48, 5120] | F16 |
| ssm_dt_bias | [48] | F32 |
| ssm_a_log | [48] | F32 |
| ssm_out.weight | [5120, 6144] | Q4_K |
| post_attention_norm.weight | [5120] | F32 |
| ffn_gate.weight | [17408, 5120] | Q4_K |
| ffn_up.weight | [17408, 5120] | Q4_K |
| ffn_down.weight | [5120, 17408] | Q4_K |

### Shared:
| Tensor | Shape | Type |
|--------|-------|------|
| token_embd.weight | [248320, 5120] | Q4_K (mmap CPU) |
| output_norm.weight | [5120] | F32 |
| output.weight | [248320, 5120] | Q6_K |

## Files to Change

1. **include/mono27b_config.h** - Add SSM/attention constants, runtime state structs
2. **include/mono27b_format.h** - Add weight entry roles
3. **src/mono27b_executor.cu** - Full rewrite: mega kernels for attn/SSM/output layers
4. **src/mono27b_chat.cpp** - Full rewrite: direct GGUF loading + decode loop
5. **CMakeLists.txt** - standalone, link cudart and cublas

## Implementation Order
1. Write plan (this file)
2. Update config/format headers with constants and state structs
3. Implement executor CUDA kernels (mega kernels for attn layer, SSM layer, output)
4. Implement chat.cpp with GGUF loader and decode loop
5. Build and test
