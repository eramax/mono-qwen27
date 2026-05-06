# Llama.cpp Alignment Optimization Progress

## Current Performance
- **Mono27B**: 26.30 tok/s (38 ms/tok)
- **llama.cpp**: 39.07 tok/s (25.6 ms/tok)
- **Gap**: 1.49x slower
- **Target**: 60 tok/s (2.28x improvement from current)

## Optimizations Completed

### 1. Q4_K Multi-Warp Kernel (+2.7%)
- **Change**: Reorganized k_q4k_mv_q8_dp4a to use 2D thread organization
- **From**: 1D block (128 threads) → **To**: 2D block (32 threads × 4 warps)
- **Algorithm**: Implemented VDR (Vector Dot Ratio) = 2 optimization
- **Result**: 25.6 → 26.3 tok/s
- **Technique**:
  - Each thread group processes blocks in parallel
  - Multiple warps coordinate work on same output row
  - Warp-level reduction followed by cross-warp reduction using shared memory

### Attempted Optimizations (Reverted)
- **Q5_K Multi-Warp**: Ported same pattern but loop structure didn't map correctly
- **Q6_K Multi-Warp**: 32 iqs values vs 16 for Q4_K causes iteration pattern mismatch
- **IQ4_XS Multi-Warp**: 8 iqs values requires different VDR calculation

## Performance Analysis

### Consistent 1.49x Gap Across All Matvecs
| Operation | Mono27B (ms) | llama.cpp (ms) | Gap |
|-----------|------------|---------------|-----|
| ssmo (Q4_K) | 11.64 | 7.83 | 1.49x |
| fg+fu | 8.22 | 5.54 | 1.48x |
| fd | 5.70 | 3.84 | 1.48x |
| re-g | 6.25 | 4.21 | 1.49x |

### Insight
The **systematic 1.49x gap** across different quantization types and kernel sizes suggests:
1. Not a single-kernel issue
2. Could be fundamental algorithm difference
3. Might be memory access pattern inefficiency
4. Register pressure or cache behavior problem
5. Instruction-level parallelism difference

## Next Steps to Explore

### High Priority
1. **Profile Actual Execution** (requires nsys/ncu):
   - Measure L2 cache hit rate
   - Check register pressure
   - Analyze memory bandwidth utilization
   - Identify stall reasons

2. **Investigate llama.cpp's Actual Implementation**:
   - Check if they use different reduction strategies
   - Analyze their shared memory usage
   - Look for instruction-level parallelism tricks
   - Check tensor core usage for larger operations

### Medium Priority
1. **Optimize Other Quantization Types Correctly**:
   - Fix Q5_K/Q6_K loop patterns for multi-warp approach
   - Consider VDR values specific to each quantization type
   - May yield 1-2% improvement per type

2. **Kernel Fusion**:
   - Combine matvec + elementwise operations
   - Could save some kernel launch overhead
   - Potential: 1-3% improvement

3. **Weight Layout Reorganization**:
   - Rearrange Q4_K blocks for better cache locality
   - Potential: 5-10% improvement if successful
   - High implementation risk

### Low Priority (Diminishing Returns)
1. **More thread scheduling tweaks**:
   - 4 warps seems optimal for single-token case
   - 8 warps regressed performance
   - 1 warp would be worse

2. **Launch overhead reduction**:
   - Already batched RMS norms (48→1 launches)
   - Multi-row kernels would help but adds complexity
   - Potential: 0.5-1% improvement

## Key Files Modified
- `src/mono27b_executor.cu`:
  - k_q4k_mv_q8_dp4a: Multi-warp implementation with VDR=2
  - l_mv_q8_on: Updated Q4_K kernel launch to use dim3(32, 4)

## Conclusion

The llama.cpp-style multi-warp kernel reorganization is correctly implemented for Q4_K and provides a 2.7% improvement. However, the fundamental 1.49x performance gap persists, suggesting the bottleneck is not thread organization but something deeper like:
- Instruction efficiency differences
- Memory bandwidth utilization
- Cache behavior differences
- Algorithm-level differences

Further optimization likely requires:
1. Hardware-level profiling to identify exact bottlenecks
2. Direct comparison of llama.cpp's actual kernel code
3. Potentially different algorithms for larger matrics (tensor cores?)

The current approach of kernel reorganization is showing diminishing returns. A breakthrough would likely require identifying and addressing the root cause of the systematic 1.49x gap.
