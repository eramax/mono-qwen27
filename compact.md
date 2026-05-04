Goal
- Fix mono27b CUDA kernels and achieve bit-exact parity with llama.cpp for Qwen3.5-27B inference without runtime dependency on llama.cpp.
Constraints & Preferences
- Do not link or depend on llama.cpp at runtime; copy reference logic/functions directly into mono27b.
- Add standalone tests to compare kernel outputs against reference formulas.
- Preserve exact file paths and identifiers from the codebase.
Progress
Done
- Read parity plan: /mnt/mydata/projects2/mono27b/docs/superpowers/plans/2026-05-04-mono27b-llama-parity.md
- Audited mono27b_executor.cu, qwen35.cpp, gated_delta_net.cu, ssm-conv.cu, norm.cu, verify_deltanet.py
- Fixed L2 norm formula bug: k_l2_norm_g used 1/sqrt(sum_sq + eps); corrected to rsqrtf(fmaxf(sum_sq, eps*eps)) to match ggml reference (debug/verify/test_kernels.cu verifies)
- Fixed SSM gated norm buffer overflow: h2 is 5120 floats; gated norm writes 6144 floats. Changed intermediate buffer to kb (17408 floats) in the SSM path
- Fixed Python GQA indexing bug: verify_deltanet.py used r_idx % ng; corrected to r_idx // (dr // ng)
- Built project successfully (cmake --build build) and ran new kernel tests (L2 norm PASS 1.49e-08, RMS norm PASS 1.19e-07)
- Verified model GGUF has n_kv = 0 (no metadata); standard Qwen epsilon is 1e-6f
In Progress
- Root-causing remaining generation-step divergence in conv_output_silu (diff ~8.176) despite conv1d math appearing correct
Blocked
- Cannot run live llama-debug reference to compare exact generation-step intermediates without building/running llama.cpp
Key Decisions
- Keep MONO27B_RMS_EPS = 1e-6f because Qwen3.5 standard uses 1e-6 and the model has no GGUF metadata override
- Reuse existing scratch buffer kb for the 6144-element gated-norm intermediate rather than growing the work buffer layout
- Copy reference formulas (not full ggml tensor system) into standalone C++ tests for validation
Next Steps
- Run end-to-end verification (make verify) to measure impact of L2-norm and buffer-overflow fixes on generation parity
- If divergence persists, instrument k_ssm_conv1d_u and k_deltanet with per-element dumps to isolate the first diverging channel/rank
- Consider porting gated_delta_net_cuda warp-accumulation order from llama.cpp if tiny rounding differences compound across 64 recurrent layers
- Add a standalone C++/CUDA test for k_ssm_conv1d_u against a CPU reference that mimics ssm_conv_f32
Critical Context
- conv_output_silu divergence (8.176) appears on generation step, not prompt step, implying stateful amplification
- h2 buffer aliasing with qb during gated norm was mathematically self-consistent but dangerous; now fixed
- Reference l2_norm_f32 uses rsqrtf(fmaxf(tmp, eps*eps)) matching PyTorch F.normalize; our old kernel used 1/sqrt(sum_sq + eps) which is wrong for small-norm vectors
- k_deltanet state indexing and gate/beta formulas match reference; GQA indexing fixed earlier to r_idx / (dr/ng)
- Model GGUF at /home/emo/Downloads/test_models/models/Qwen3.6-27B-UD-Q4_K_XL.gguf has zero KV metadata entries
Relevant Files
- /mnt/mydata/projects2/mono27b/src/mono27b_executor.cu: main inference kernels; fixed k_l2_norm_g and SSM gated-norm buffer usage
- /mnt/mydata/projects2/mono27b/include/mono27b_config.h: defines MONO27B_RMS_EPS = 1e-6f
- /mnt/mydata/projects2/mono27b/debug/verify/test_kernels.cu: new standalone CUDA test comparing L2/RMS norm against reference formulas
- /mnt/mydata/projects2/mono27b/debug/verify/verify_deltanet.py: updated GQA indexing fix
- /mnt/mydata/projects2/mono27b/ref/llama.cpp/ggml/src/ggml-cuda/norm.cu: reference L2 norm (l2_norm_f32) and RMS norm (rms_norm_f32) implementations
- /mnt/mydata/projects2/mono27b/ref/llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu: reference ssm_conv_f32 kernel (no F16 path)
- /mnt/mydata/projects2/mono27b/ref/llama.cpp/src/models/qwen35.cpp: reference graph construction for Qwen3.5
- /mnt/mydata/projects2/mono27b/docs/superpowers/plans/2026-05-04-mono27b-llama-parity.md: parity tracking plan with replay diff table