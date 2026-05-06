
# Speed Optimization Ideas

## Completed (18.4 → 25.5 tok/s greedy, +38.6%)
- [x] Warp shuffle reduction (block_reduce_sum) — replaces shared memory tree in 4 matvec kernels
- [x] Batched RMS norm — fused 48+28 individual launches into 2 per layer
- [x] GPU argmax — avoids 1MB D2H copy in greedy mode
- [x] Removed unnecessary cudaDeviceSynchronize in LM head

## High Impact (estimated +30-50%)
- [ ] **CUDA Graphs**: Capture the entire decode_step as a graph and replay per token. 
  - Challenge: dynamic `pos` parameter changes each step
  - Solution: Store `pos` in device memory, update via cudaMemcpy before replay
  - This would eliminate ALL host-side launch overhead (~2000 launches × 5µs = 10ms)
- [ ] **MMQ path for large outputs**: For FFN (17408 output) and LM head (248320), use matrix-matrix quantized kernels
  - These read the input multiple times but with better parallelism

## Medium Impact (estimated +10-20%)
- [ ] **Fuse elementwise ops**: silu+mul, softplus+mul can be single kernels
- [ ] **Fuse quantize+matvec**: For repeated patterns like MV(L.wqkv, h2, sb), do quantize+matvec in one kernel
- [ ] **Reduce attention kernel shared memory**: k_attn_parallel allocates max_ctx floats in shared memory

## Low Impact / Research
- [ ] Use tensor cores (WMMA) for dequant→FP16→WMMA matmul path
- [ ] Paged KV cache to reduce memory footprint
- [ ] Flash attention for longer contexts
