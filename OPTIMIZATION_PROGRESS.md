# Current Optimization Status

## Performance Metrics
- **Current**: 25.6 tok/s (39.2 ms/tok)
- **Baseline**: 18.4 tok/s (54.3 ms/tok)
- **Target**: 60 tok/s (16.7 ms/tok)
- **Improvement from baseline**: +39% (18.4 → 25.6 tok/s)
- **Gap to target**: 2.34x speedup needed

## Bottleneck Analysis
| Operation | Time/Token | % | Speedup Needed |
|-----------|----------|---|---|
| ssmo (SSM output matvec) | 11.64 ms | 31.8% | 3.9x |
| fg+fu (FFN gate+up) | 8.27 ms | 22.6% | 2.9x |
| fd (FFN down) | 5.30 ms | 14.5% | 1.8x |
| re-g (SSM wqkv_gate) | 6.08 ms | 16.6% | 2.0x |
| Other | 8.16 ms | 14.5% | 1.0x |
| **Total** | **39.45 ms** | **100%** | **2.35x** |

## Optimizations Implemented This Session
1. **Fused copy + residual add kernel** (k_elem_copy_add)
   - Saves one kernel launch for SSM residual paths
   - Minimal performance impact (~0 ms improvement)
   
2. **Fused RMS norm + gated multiply** (k_rms_norm_mulw_mul_batched)
   - Combines two small kernels into one
   - Reduces launches from 96 to 48 for SSM gating
   - Minimal performance impact (~0 ms improvement)

## Previous Session Optimizations (Already Included)
1. **Batched RMS norm** (19.6 → 25.5 tok/s, +30%)
   - Reduced kernel launches from 4659 to 1695 per token
   - Single biggest win

2. **Warp shuffle reduction** (+11%)
   - Replaced shared memory tree reduction with warp shuffle
   
3. **GPU argmax** for greedy sampling

4. **Redundant wqkv_gate elimination** (+2.8%)
   - Reused buffer instead of recomputing

## Key Insights
- **Kernel launch overhead** is not the main bottleneck (already batched RMS norms)
- **Memory bandwidth utilization** gap is 10x: 70 GB/s vs 774 GB/s microbenchmark
- **Single-row kernels are optimal** for this workload (multi-row kernels regressed)
- **Element-wise kernel fusion** has minimal impact (those operations are already fast)

## Remaining Challenges
1. **Large matvecs** (ssmo, fg+fu, fd) use 70 GB/s vs potential 900 GB/s
   - Root cause may be: L2 cache eviction, poor memory coalescing, or algorithmic limitations
   
2. **Need 2.35x speedup** from current state
   - Even 3x faster matvecs would only give ~42 tok/s (still short of 60 tok/s target)
   - May require fundamental algorithmic changes or different quantization

3. **Tensor core approach** challenging for matvecs
   - WMMA is optimized for 16x16+ matrix ops
   - Matvecs are inherently 1xN operations
   - Previous multi-row attempts regressed performance

## Next Steps to Consider
1. **Profile with NVIDIA nsys/ncu** to identify exact stall reasons
2. **Implement weight reorganization** for better memory coalescing
3. **Try alternative quantization** schemes that compress better or compute faster
4. **Evaluate different execution models** (e.g., dynamic batching across tokens)
5. **Implement IQ4_XS tensor core variant** if time permits

## Feasibility Assessment
Reaching 60 tok/s appears extremely challenging without:
- Significantly better algorithms for matvec computation
- Different quantization schemes
- Fundamental architectural changes
- Or access to higher-bandwidth GPUs

Current improvements have focused on kernel launch overhead, which was successfully addressed through batching. The remaining gap is computation speed itself, which is harder to improve without deeper algorithm changes.
