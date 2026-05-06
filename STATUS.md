# mono27b — Speed Optimization Status

**Last updated:** 2026-05-06  
**Current best:** 25.5 tok/s (greedy), 24.5 tok/s (sampling)  
**Target:** 39.3 tok/s (llama.cpp reference)  
**Gap:** 13.8 ms/tok GPU time (~14.7 ms/tok wall clock)

---

## What's Done ✅

| # | Optimization | Impact | Status |
|---|-------------|--------|--------|
| 1 | Warp shuffle reduction in 4 matvec kernels | +11% | ✅ Kept |
| 2 | Batched RMS norm (2304→2 launches/layer) | +29% | ✅ Kept |
| 3 | GPU argmax (avoids 1MB D2H in greedy) | Greedy boost | ✅ Kept |
| 4 | Q8_1 input caching (skip re-quantize) | ~+0.5% | ✅ Kept |
| 5 | CUDA graph capture for decode_step | +2% | ✅ Kept |
| 6 | Timing instrumentation (`make compare-perf`) | — | ✅ Infrastructure |

**Cumulative speedup: 18.4 → 25.5 tok/s (+38%)**

---

## Current Bottlenecks (from timing data)

### Top 3 GPU Time Consumers

1. **SSM output matvec (`ssmo`)** — 11.7 ms/tok (30.9%)
   - 48 SSM layers × 0.24 ms each
   - 10× slower in pipeline than in isolation
   - **Action:** Rewrite with multi-row blocks + shared Q8_1 cache

2. **SSM wqkv_gate matvec (`re-g`)** — 7.5 ms/tok (19.8%)
   - 48 SSM layers × 0.16 ms each
   - Q5_K path may be suboptimal
   - **Action:** Port llama.cpp's Q5_K vec_dot

3. **Inter-layer overhead (`pre`)** — 11.1 ms/tok (29.4%)
   - NOT a single kernel; captures pipeline bubbles between layers
   - ~1700 kernel launches per token, many tiny (<10 µs)
   - **Action:** Persistent kernel or kernel fusion per layer

### Remaining Sync Points

| Location | Count | Impact |
|----------|-------|--------|
| After all layers (`cudaDeviceSynchronize`) | 1/tok | ~0.5 ms |
| After LM head quantize+matvec | 1/tok | ~0.3 ms |
| Timing commit (`cudaEventSynchronize`) | 1/tok | ~0.5 ms (timing builds only) |

---

## In Progress 🔄

| Task | Status | Notes |
|------|--------|-------|
| Fused gate+up kernel | ❌ Discarded | 0.8% improvement — FFN not the bottleneck |
| Attention shared memory reduction | ❌ Discarded | No improvement — attention only 0.5 ms/tok |
| Multi-row matvec x4 + shared Q8_1 | ❌ Discarded | -3.7% regression — overhead exceeds benefit |
| GPU top-k (naive sequential) | ❌ Discarded | -50% regression — need parallel heap approach |
| Timing comparison script (`make compare-perf`) | ✅ Done | See `scripts/compare_perf.py` |

---

## TODO List 📋

### High Priority

- [ ] **Profile with Nsight Systems** — identify exact pipeline bubbles and idle periods
- [ ] **Rewrite `ssmo` matvec kernel** — multi-row per block (4 rows) + shared Q8_1 cache
- [ ] **Port llama.cpp Q5_K vec_dot** — compare and adopt their instruction sequence
- [ ] **Fuse SSM elementwise ops** — silu + rms_norm + mul into single kernel per layer

### Medium Priority

- [ ] **GPU Top-K partial sort** — avoid 1MB D2H copy for sampling mode
- [ ] **Persistent kernel experiment** — single kernel processes entire SSM layer
- [ ] **MMQ path for LM head** — 248320 output rows, large enough for matrix-matrix kernel
- [ ] **Remove output path syncs** — replace `cudaDeviceSynchronize` with async event-based sync

### Low Priority

- [ ] **Investigate Q8_1 cache bank conflicts** — shared memory layout optimization
- [ ] **Try 256-thread elementwise kernels** — may improve occupancy for small arrays
- [ ] **Tensor core path (WMMA)** — for future GPUs with better FP16/INT8 support

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Kernel launches / token | ~1700 (was ~4659 before batched RMS) |
| Matvec kernels / token | 316 (48 SSM × 5 + 16 attn × 3 + 1 LM head) |
| GPU bandwidth (isolation) | 774 GB/s (83% peak) |
| GPU bandwidth (pipeline) | ~70 GB/s (7% peak) |
| CPU overhead / token | ~2.5 ms (6% of wall clock) |

---

## Reference Commands

```bash
# Build optimized (no timing)
make build-fast

# Build with timing instrumentation
make build-timing

# Run full comparison vs llama.cpp
make compare-perf

# Quick test
make quick-test

# Custom run
make test PROMPT="Hello world" GEN=50 CTX=4096
```

---

## Files Modified in This Session

- `src/mono27b_executor.cu` — Main optimization work (matvec kernels, batched RMS, CUDA graphs, timing)
- `src/mono27b_chat.cpp` — GPU argmax integration, timing print
- `include/mono27b_config.h` — Timing struct additions
- `CMakeLists.txt` — `MONO27B_TIMING` option
- `Makefile` — `compare-perf`, `build-fast`, `build-timing` targets
- `scripts/compare_perf.py` — Performance comparison parser
- `ANALYSIS.md` — This analysis
- `STATUS.md` — This status report
