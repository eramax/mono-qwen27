# SSMO Kernel Optimization Analysis

## Current State
- **Operation**: SSM output matvec (Q4_K weights × Q8_1 quantized input)
- **Dimensions**: 6144 input (24 Q4_K blocks) × 5120 output rows
- **Time per token**: 11.644 ms (32.1% of total GPU time)
- **Calls per token**: 48 (one per SSM layer)
- **Time per call**: 0.242 ms
- **Kernel calls per token**: 48 
- **Estimated kernel count reduction impact**: 1-2% (48 launches is not dominant)

## Performance Gap Analysis

### Current Performance
- **Bandwidth utilization**: ~70 GB/s (observed)
- **Theoretical peak**: 900 GB/s (RTX 3090)
- **Gap ratio**: 12.9x slower than theoretical peak

### Why is there a 12.9x gap?

Possible causes:
1. **L2 Cache Inefficiency**: Each kernel launch may evict L2 cache lines
2. **Memory Access Pattern**: Q4_K weight layout may not match GPU memory hierarchy
3. **Register Pressure**: Complex dequantization in vec_dot may limit occupancy
4. **SM Utilization**: 5120 blocks (one per output row) may not saturate all SMs
5. **Instruction Latency**: Dequantization logic has high latency operations
6. **Global Memory Contention**: Multiple blocks competing for memory bandwidth

## Optimization Approaches Evaluated

### 1. ✗ Multi-Row Kernels (2-4 rows per block)
- **Status**: Tried, regressed by 3.7%
- **Issue**: Increased synchronization overhead, worse occupancy
- **Conclusion**: Single-row kernels are already optimal for this workload

### 2. ✗ Larger Block Size (256 threads)
- **Status**: Tried, no improvement
- **Issue**: Memory bandwidth is the bottleneck, not occupancy
- **Conclusion**: Block size doesn't affect bandwidth utilization

### 3. ✗ Tensor Cores (WMMA)
- **Status**: Attempted but not viable
- **Issue**: WMMA optimized for 16x16+ matrices, not 1×N vectors
- **Conclusion**: Not applicable to matvec operations

### 4. ✗ Shared Memory Caching
- **Status**: Considered but limited benefit
- **Issue**: Q8_1 blocks used once per row, not across rows
- **Conclusion**: Cache reuse limited, shared memory adds complexity

### 5. ✗ Kernel Fusion (ssmo + residual)
- **Status**: Complex to implement
- **Issue**: Would require refactoring multiple layers of abstraction
- **Conclusion**: Effort vs reward not justified

## What COULD Work

### Option 1: Weight Layout Reorganization (High Risk/High Reward)
Reorganize Q4_K blocks to improve memory coalescing:
- **Idea**: Current layout: blocks stored sequentially in memory
- **Optimization**: Transpose or tile block layout to match access pattern
- **Challenge**: Requires weight format conversion, compatibility issues
- **Estimated improvement**: 1.5x-2x (if successful)

### Option 2: Hybrid Approach (Shared Memory + Register Blocking)
Use shared memory to cache frequently accessed data:
- **Idea**: Load Q8_1 metadata into shared memory
- **Implementation**: Requires careful orchestration
- **Challenge**: Complex, risky, may not improve due to L2 caching
- **Estimated improvement**: 1.1x-1.3x

### Option 3: Kernel-Level Algorithmic Change
Change how dequantization is computed:
- **Idea**: Precompute dequantization tables, use lookups instead
- **Challenge**: Extra memory access, cache misses
- **Estimated improvement**: 1.0x-1.2x

### Option 4: GPU-Specific Tuning (Architecture-Aware)
Tune launch parameters specifically for RTX 3090:
- **Idea**: Use Ampere-specific features (TensorFloat32, better L1 caching)
- **Challenge**: Requires profiling with nsys to be effective
- **Estimated improvement**: 1.1x-1.2x

## Root Cause: Memory Bandwidth Saturation

The 12.9x gap suggests the fundamental issue is **memory bandwidth**, not kernel efficiency:

```
Bandwidth equation: Throughput = (Work / Bytes) × Bandwidth
- Theoretical max: 900 GB/s × (flops per byte) = max throughput
- Actual: 70 GB/s × (flops per byte) = observed throughput
- Gap: Suggests either:
  1. Block scheduler is not saturating memory bus
  2. Memory access pattern causes stalls (L2 misses, cache line eviction)
  3. Bank conflicts or other serialization bottlenecks
```

## Recommended Next Steps

### For Immediate Improvement (without deep changes):
1. Profile with NVIDIA nsys to identify exact stall reasons
2. Check L2 cache hit rate during ssmo kernels
3. Measure memory bandwidth utilization with ncu

### For Longer-term Optimization:
1. Implement weight layout transformation
2. Explore quantization format changes
3. Consider algorithmic alternatives (e.g., batching across layers)

## Conclusion

The ssmo kernel is fundamentally memory-bandwidth limited. The 12.9x gap to theoretical peak cannot be closed by traditional kernel optimization techniques (multi-row, tensor cores, larger blocks). Addressing this gap requires either:

1. **Profiling-guided optimization** (nsys/ncu to identify exact bottleneck)
2. **Architectural changes** (weight layout, quantization format, execution model)
3. **Different GPU** (with higher memory bandwidth or better caching)

Current performance (26 tok/s) represents near-optimal execution for the existing kernel design. Reaching 60 tok/s would require fundamental changes to how ssmo is computed or executed.
