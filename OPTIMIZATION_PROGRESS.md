# Current Optimization Status

## Performance Metrics
- **Current**: 26.3 tok/s (38.0 ms/tok)
- **Baseline**: 18.4 tok/s (54.3 ms/tok)
- **Target**: 60 tok/s (16.7 ms/tok)
- **Improvement from baseline**: +43% (18.4 → 26.3 tok/s)
- **Gap to target**: 2.28x speedup needed
- **Latest gain**: Q4_K multi-warp optimization (+2.7%)

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

### Latest: Q4_K Multi-Warp Kernel (+2.7%)
- **Implementation**: Reorganized k_q4k_mv_q8_dp4a kernel to follow llama.cpp pattern
- **Change**: 1D thread organization (128 threads) → 2D (32×4 warps)
- **Algorithm**: VDR (Vector Dot Ratio) = 2 for better ILP
- **Result**: 25.6 → 26.3 tok/s
- **How it works**:
  - Multiple warps coordinate work on single output row
  - Each thread group processes blocks in parallel
  - Warp-level shuffle reduction + cross-warp shared memory reduction

### Previous: Fused Kernels (minimal impact)
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

### Systematic 1.49x Performance Gap
- **Observation**: All matvec operations (ssmo, fg+fu, fd, re-g) are ~1.49x slower than llama.cpp
- **Implication**: Not a single-kernel issue, but systematic difference
- **Possible causes**:
  - Instruction-level parallelism differences
  - Memory access pattern inefficiency
  - Register pressure or L2 cache behavior
  - Fundamental algorithm differences (tensor cores?)

### Performance Gap Analysis
- **Q4_K optimization impact**: Only +2.7% despite matching llama.cpp's thread organization
- **Q5_K/Q6_K attempts**: Loop iteration patterns don't map well to multi-warp approach
- **Conclusion**: Thread organization is not the primary bottleneck

### Speedup Requirements
1. **Current path**: Need 2.28x speedup from current 26.3 tok/s
2. **If matvecs were 1.49x faster**: Would achieve ~39 tok/s (still 1.5x short of 60 tok/s target)
3. **Implication**: Even perfect matvec optimization wouldn't reach 60 tok/s
   - Would need tensor core kernels, algorithmic changes, or different quantization strategies

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

## Latest Session: Q4_K Multi-Warp Kernel Optimization

### Achievement
- Implemented llama.cpp-style multi-warp kernel organization for Q4_K
- Performance gain: 25.6 → 26.3 tok/s (+2.7%)
- Thread reorganization: 1D (128) → 2D (32×4 warps)
- Added VDR=2 optimization for better instruction-level parallelism

### Key Findings
1. **Systematic 1.49x Gap**: All matvec operations are consistently ~1.49x slower than llama.cpp
2. **Diminishing Returns**: Kernel structure optimization yielded only 2.7% improvement
3. **Root Cause Unknown**: The systematic nature of the gap suggests something deeper than thread organization
   - Possibly memory access patterns
   - Possibly instruction efficiency
   - Possibly different algorithm (tensor cores?)

### Attempts Made
- Q5_K multi-warp optimization: Reverted (loop iteration patterns incompatible)
- Q6_K multi-warp optimization: Reverted (32 iqs values don't map to pattern)
- IQ4_XS multi-warp optimization: Reverted

### Path Forward
To reach 60 tok/s requires 2.28x improvement from current state. This is extremely challenging because:
1. Even if matvecs were 1.49x faster (matching llama.cpp), we'd only reach ~39 tok/s
2. Would still need additional 1.5x speedup
3. Likely requires tensor core kernels or fundamental algorithm changes

### Conclusion
The Q4_K optimization is working and correct, but represents diminishing returns. The systematic 1.49x gap indicates the bottleneck is not thread organization but something more fundamental that would require profiling tools or direct access to llama.cpp's source to understand and address.
