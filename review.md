Here’s a detailed review of your CUDA executor and a set of optimisation ideas—focused on increasing throughput on an RTX 3090 (Ampere, SM 8.6). The code already contains many good practices (CUDA graphs, warp‑level reductions, batched RMS norm, multi‑stream QKV, fused gate‑and‑WO quantisation). The suggestions below aim to further reduce kernel launch count, memory traffic, and synchronisation overhead, while respecting the numerical properties of the models you run.

---

## 1. Fuse Q‑head RoPE directly into the online‑attention kernel

**Current path:**
1. De‑interleave Q and gate → `kb` / `qb`
2. `k_mrope` rotates `Q` in‑place → reads + writes `qb`
3. `k_attn_decode_online` reads `qb` (rotated) for every key position

**Observation:** In your attention kernel, each thread processes elements of `Q` at indices `t, t+blockDim, t+2*blockDim, …` . For `head_dim = 256` and `blockDim = 128`, a thread naturally owns exactly the two indices of each RoPE pair (e.g., `d` and `d+128`). Therefore you can apply the rotation on‑the‑fly when computing the dot products, without ever writing the rotated `Q` back to memory.

**What to change:**
- Pass the *un‑rotated* `Q` to the attention kernel (i.e. skip `k_mrope`).
- Inside the kernel’s `for (p = 0; p < kv_len; ++p)` loop, when accumulating `dot_local += qr[i] * kr[i]`, load both paired components for the current `i` and rotate them with the precomputed `cos`/`sin`. This costs a few extra flops but eliminates:
  - one kernel launch (`k_mrope`)
  - one full read + write of `qb` (size `n_heads * head_dim * 4` bytes)

**Effect:** Saves memory bandwidth and one graph node per attention layer.

---

## 2. Eliminate the redundant gate‑copy in SSM layers

**Current code in SSM path:**
```cpp
MV_PAIR(L.wqkv, L.wqkv_gate, h2, sb, gb);
cudaMemcpyAsync(fb, gb, MONO27B_SSM_D_INNER * sizeof(float), …);
```
Later, `fb` (the saved gate) is used by `k_elem_silu` and then fed into the batched RMS‑norm‑mul kernel.

**Fix:** Let `MV_PAIR` write the gate directly into `fb`:
```cpp
MV_PAIR(L.wqkv, L.wqkv_gate, h2, sb, fb);   // y2 = fb
```
No device‑to‑device copy needed. `fb` is already a free scratch buffer at that point.

---

## 3. Fuse the SSM parameter computation (β and Δ) into one kernel

After the paired matvecs you currently launch three elementwise kernels:
- `k_elem_sigmoid` on `kb` (β)
- `k_elem_softplus` + bias on `qb` (α)
- `k_elem_mul` on `qb` × `ssm_a_log`

All operate on the same small vector of size `MONO27B_SSM_DT_RANK` (likely 128–512). Launching three kernels is wasteful.

**Suggestion:** Write a single kernel that reads the raw `beta_raw` (`kb`), `alpha_raw` (`qb`), the bias, and the `a_log` array, and writes the final `beta` and `delta_t` back to `kb` / `qb`. This reduces kernel launch overhead and avoids redundant global memory traffic for such small arrays. A simple block of 128 threads can cover the whole rank in one launch.

---

## 4. Fuse silu activation into the Conv1D kernel

**Currently:**
```cpp
k_ssm_conv1d_u(…, sb);        // writes linear conv result to sb
k_elem_silu(sb, MONO27B_SSM_CONV_CH);
```

**Fusion:** After computing the linear combination inside `k_ssm_conv1d_u`, apply `output = input / (1.f + expf(-input))` before writing to `sb`. This eliminates the separate `k_elem_silu` launch and one full read/write of `sb` (size `conv_ch`). The Conv1D kernel already operates per‑channel; adding a `silu` is trivial and cost‑free.

---

## 5. Merge `ssm_out` matvec with the residual add

In the SSM branch, after the gate+norm step you do:
```cpp
MV(L.ssm_out, kb, sb);
k_elem_copy_add<<<…>>>(sb, h, h2, MONO27B_TARGET_HIDDEN);
```

Your infrastructure already supports a residual‑add inside the matvec (see the `residual` parameter in `l_mv_q8_on`). Quantise `kb` (size `SSM_D_INNER`) into Q8_1 blocks, then call:
```cpp
l_mv_q8_on(L.ssm_out.ptr, L.ssm_out.ggml_type, …, …, h2, h, 0);
```
This writes `ssm_out + residual` directly into `h2`, removing the extra `copy_add` kernel.

---

## 6. Fuse QKV matvecs for attention into a single kernel

You already attempted concurrency with a second stream, but synchronisation overhead may erode the gain. A more robust approach is to write a dedicated QKV matvec kernel that reads the **same quantised input once** and performs three independent dot products per thread block, writing to three separate output buffers (`Q_out`, `K_out`, `V_out`).

- **Benefits:**
  - One kernel launch instead of three.
  - The quantised input vector (Q8_1 blocks) is streamed through the SMs only once.
  - Better occupancy: the thread block can interleave work to hide latency.
- **Complexity:** Manageable – the kernel would be a variant of `k_q4k_mv_q8_dp4a` that computes three sums instead of one, using the same VDR loop.

Given that Wq, Wk, and Wv often share the same quantisation type and dimensions, this is a high‑impact fusion.

---

## 7. Optimise the online softmax attention kernel

- **Vectorised memory access:** Use `float4` loads when reading `K` and `V` from the cache. This improves L1 bandwidth.
- **Register tiling for V accumulation:** Instead of updating `out_acc` per position with a single scalar, accumulate a small register array of `float4` (e.g., 4 elements) and reduce at the end. This trades a few registers for higher ILP.
- **Warp‑level softmax statistics:** Your current scheme uses `__shfl_xor_sync` inside each position’s dot product, then smem for inter‑warp reduction. That’s already good.

No algorithmic change can avoid the per‑position barrier, but you can reduce the cost of each iteration.

---

## 8. Use `__ldg` and L1 cache hints for read‑only data

Throughout all matvec, RMS‑norm, and elementwise kernels, load weight tensors and input vectors that are never written back with `__ldg()`. This forces L1‑cached reads on SM 8.6 and can improve effective bandwidth. Example:
```cpp
const float v = __ldg(x + i);
```
It’s low effort and helps many small kernels.

---

## 9. Increase occupancy of the Q4_K matvec kernel

The kernel `k_q4k_mv_q8_dp4a` currently uses 4 warps (128 threads). On 3090, you can afford up to 8–12 warps without excessive register pressure. Increasing the block size to 256 threads (8 warps) would allow each block to process more blocks per iteration (VDR * NWARPS * 32 / QI = 2 * 8 * 32 / 32 = 16 blocks per iter instead of 8), reducing the loop trip count and improving SM utilisation. You’ll need to adjust the shared memory reduction accordingly. Test the trade‑off between occupancy and register spilling; use `--ptxas-options=-v` to monitor.

---

## 10. Micro‑fusions and cleanup

- **De‑interleave Q and gate**: If item 6 (fused QKV matvec) is adopted, the interleaved Q/gate output problem disappears for attention layers because we can write the matvec outputs directly in de‑interleaved layout using a smart row mapping. Alternatively, keep the current approach but write the de‑interleaved output directly during the matvec (requires a small indexing trick in the matvec kernel).
- **RMS‑norm + residual**: After the FFN down‑projection you do `MV(L.ffn_down, fb, h); k_elem_add(h, h2, …)`. Again, the matvec can accept the residual `h2` and write the sum directly to `h`, eliminating the elementwise add. (This requires extending `l_mv_quant`/`l_mv_q8` to accept a residual pointer; already partially done).
- **F16 intermediate activations**: Consider storing the hidden state `h` in half precision across layers. The 3090 has full‑rate FP16 arithmetic, and this halves the register and memory footprint of all elementwise ops and RMS norms. You would need to insert `__half2float` conversions only at the boundaries (matvec input, attention). This can be a large win for long sequences, but it requires thorough numerical validation.

---

## 11. Streamline the graph capture

Your CUDA graph captures the entire layer loop. That’s excellent. After applying the fusions above, the number of graph nodes will drop, making the graph replay even faster. Ensure that all host‑side updates (`cudaMemcpyToSymbol` for `g_mrope_pos`, `g_kv_pos`, `g_tok_id`) happen before the graph launch, as you already do. Consider using **graph launch with scalar arguments** if these positions change every step – but your current symbol approach works fine.

---

## Summary of impact

| Optimisation                             | Kernel launches saved per layer | Memory traffic saved              |
|------------------------------------------|---------------------------------|-----------------------------------|
| Fuse Q‑RoPE into attention               | 1 (attention layers)           | Q head tensor read + write        |
| Remove SSM gate copy                     | 0 (just a copy removed)        | 1× `D_INNER` bytes                |
| Fuse SSM β/Δ computation                 | 2 elementwise kernels          | small, but reduces launch overhead|
| Fuse Conv1D + silu                       | 1 kernel                       | `CONV_CH` read + write            |
| Fuse ssm_out matvec + residual           | 1 kernel                       | `HIDDEN` read + write             |
| Fused QKV matvec                         | 2 kernels                      | shared input vector read once     |
| Other residual adds (FFN down)           | 1 per layer                    | `HIDDEN` r/w                      |

For a 24‑layer model with alternating attention/SSM, the total reduction in graph nodes could be **>30%**, leading to a measurable throughput improvement. The largest gains will come from the fused QKV matvec and the Q‑RoPE/attention fusion because they touch large tensors.

Start with the low‑risk items (gate copy, Conv1D+silu, SSM param fusion), then progressively integrate the more complex fusions. Most changes are local and can be validated layer‑by‑layer with a golden reference.

=========

I’ve reviewed the full codebase—the CUDA executor, the CPU‑side tokenizer, GGUF parser, blob functions, and the chat driver. Below are concrete ways to **shrink the code**, **make it faster**, and **simplify maintenance**, all tuned for an RTX 3090.

---

## 1. CUDA executor: deeper kernel fusion (further speed‑ups)

Your existing fusions (batched RMS‑norm, fused gate‑and‑WO quant, online attention) are already excellent. The next level:

### a) Fuse the Q‑head RoPE into the online attention kernel

Currently you rotate `qb` with `k_mrope` and then read it again in `k_attn_decode_online`. Because each attention thread owns exactly one RoPE pair (indices `d` and `d+half`), you can apply rotation **on‑the‑fly** inside the dot‑product loop without ever writing `qb` back to memory. Remove `k_mrope` from the attention layers.

**Benefit:** Eliminates one kernel launch per attention layer and saves half the bandwidth for the Q tensor.

### b) Remove the redundant device‑to‑device copy of SSM gate

In the SSM branch you currently do:
```cpp
MV_PAIR(L.wqkv, L.wqkv_gate, h2, sb, gb);
cudaMemcpyAsync(fb, gb, ...);   // copy gate away
```
Simply let `MV_PAIR` write the gate directly to `fb`:
```cpp
MV_PAIR(L.wqkv, L.wqkv_gate, h2, sb, fb);
```
Drop the `cudaMemcpyAsync`.

**Benefit:** Saves one D2D copy and a synchronisation point.

### c) Fuse the SSM β and Δ computation into a single kernel

Three elementwise kernels (`sigmoid`, `softplus`+bias, `mul`) operate on small arrays of size `SSM_DT_RANK`. Replace them with one kernel that reads the raw inputs and writes the final `beta` and `delta_t`. Launch overhead and memory traffic are removed.

### d) Fuse silu activation into the Conv1D kernel

After `k_ssm_conv1d_u`, you call `k_elem_silu`. Inside the Conv1D kernel, apply `silu` before storing. No extra launch needed.

### e) Fuse `ssm_out` matvec with the residual add

Use `l_mv_q8_on` with the `residual` argument to write `ssm_out + residual` directly into `h2`, avoiding the later `k_elem_copy_add`.

### f) Merge QKV matvecs into one kernel

Write a single kernel that loads the quantised input vector once and computes three dot products (Q, K, V) per thread block. This can halve the reads of the large hidden state and reduce launch count. Useful because Wq, Wk, Wv often share the same quant type.

### g) Utilise `float4` loads in the online attention kernel

When streaming K and V from the cache, use `float4` to improve L1 bandwidth.

---

## 2. Sampling: move to GPU to eliminate host‑device sync and copy

Currently you copy all 248k logits to the host every token, then run CPU sampling. On the RTX 3090 you can sample directly on the device:

- Write a small CUDA kernel that does temperature scaling, top‑k, top‑p, min‑p and outputs the chosen token ID to `st->argmax_result`.
- Use `curand` for random numbers inside the kernel.
- No need to copy logits; the decode step keeps them in `work_buf`.

**Benefit:** Removes a 1 MB copy per token and the CPU sampling overhead, gaining 0.5–1.0 ms/token.

---

## 3. Tokenizer: use the standard BPE “linked‑list” algorithm

Your `bpe()` function recomputes all pairs every iteration (`O(n²)`). Instead, use the **GPT‑2 tokenizer algorithm** (like Hugging Face’s `tokenizers` library): maintain a doubly‑linked list of symbols and a priority queue (or array) of merge candidates, updating only neighbours after each merge. This reduces worst‑case complexity to near‑linear.

Additionally:
- Replace `unordered_map<string,int>` for merge ranks with a custom hash map that uses `string_view` to avoid allocations.
- Pre‑split the input into “words” using the same logic as `encode()` already does; then the BPE operates on short strings.
- Cache frequent words.

**Benefit:** Much faster encoding of large prompts, crucial for long‑context prefill.

---

## 4. Code size reduction & maintainability

### a) Wrap repeated CUDA launch patterns into helper functions

You already have `l_mv_quant`, `l_rms`, `MV` macros. Further unify elementwise launches:

```cpp
template<int BLK, typename F>
void launch_elem(int n, F kernel_func) {
    kernel_func<<<(n + BLK - 1) / BLK, BLK>>>(...);
}
```

Then replace the many `<<<…>>>` lines with calls.

### b) Use RAII for CUDA resources

Replace manual `cudaFree` and `goto fail` in `mono27b_engine_init_state` with `std::unique_ptr<void, decltype(&cudaFree)>` or a custom `CudaBuffer` class. This eliminates the error‑prone `goto fail` and reduces LOC.

### c) Split the huge `mono27b_executor.cu` into logical files

- `matvec.cu` – all dot‑product kernels and their helpers
- `attention.cu` – online attention, RoPE, KV cache
- `ssm.cu` – conv1d, deltanet, SSM helper kernels
- `norm.cu` – RMS norm, elementwise ops
- `engine.cu` – decode_step, state management

This makes the codebase easier to navigate and doesn’t hurt compilation time if you use separate compilation units.

### d) Remove the `MONO27B_TIMING` preprocessor tangling

Replace with a lightweight timing class that is always compiled but can be disabled via a runtime flag. This eliminates the `#ifdef` noise and lets you keep one clean code path.

### e) Simplify GGUF parsing

The `mono27b_gguf.cpp` is straightforward but has a huge `read_kv_value` with manual padding and skipping. You could replace the `skip_value` logic with a single `std::fseek` if the value is not needed, using pre‑computed type sizes. Or parse directly from the mmap’ed data buffer (avoid `FILE*` entirely) to reduce LOC and allocations.

### f) Unify blob and tokenizer reading

`mono27b_blob.cpp` and `mono27b_tokenizer.cpp` share a lot of low‑level `read_pod`/`read_string` patterns. Move them to a common `Mono27BReader` class that wraps either a `FILE*` or a `const uint8_t*` buffer, providing `read<T>()` and `read_string()` methods.

---

## 5. Memory and occupancy tuning

- **Q4_K matvec kernel:** Increase block size to 8 warps (256 threads) to better hide latency on the 3090. Adjust shared memory usage accordingly.
- **Quantisation scratch:** Currently `q8_scratch_max_blocks = 2048`. For the largest matvec (e.g., down projection), that’s enough; but you might want to increase it to 4096 to avoid fallback in some edge cases.
- **Use `__ldg()`** for all read‑only weight tensors to force L1 caching.

---

## 6. Example refactoring snippet: unified elementwise launch

Before:
```cpp
k_elem_silu<<<(n + 255) / 256, 256>>>(x, n);
k_elem_sigmoid<<<(n + 255) / 256, 256>>>(x, n);
```
After:
```cpp
auto launch = [](int n, auto kernel, auto... args) {
    constexpr int BLK = 256;
    kernel<<<(n + BLK - 1) / BLK, BLK>>>(args...);
};
launch(MONO27B_TARGET_HIDDEN, k_elem_silu, x, MONO27B_TARGET_HIDDEN);
launch(MONO27B_TARGET_HIDDEN, k_elem_sigmoid, x, MONO27B_TARGET_HIDDEN);
```
But better: create a template function `activate` that picks the kernel at compile time.

---

## 7. Summary of expected improvements

| Area                     | Gain                                      |
|--------------------------|-------------------------------------------|
| Fused Q‑RoPE + attention | −1 launch/layer, ~5% bandwidth            |
| SSM gate copy removal    | −0.5 ms/token                             |
| Fused SSM β/Δ + silu     | −3 launches per SSM layer                 |
| Fused QKV matvec         | −2 launches, +10% matvec throughput        |
| GPU sampling             | +1 tok/s (removes host copy)              |
| Tokenizer optimisation   | Much faster prefill                       |
| Code size reduction      | ~30% fewer lines in executor              |

Start with the low‑risk items (gate copy, fused elementwise ops), then tackle the QKV fusion and GPU sampling for the biggest speed wins. Each change can be validated independently by comparing outputs to the current version.


