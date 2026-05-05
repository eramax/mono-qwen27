# Verification Makefile for mono27b
# make verify        -> runs BOTH llama.cpp ref AND our tool, compares side-by-side
# make verify-only   -> only runs checks on existing data
# make e2e           -> end-to-end logit comparison only
# make compare-all   -> comprehensive tensor comparison (all layers, all tensors)

GGUF ?= /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
PROMPT ?= give
DEBUG ?= /tmp/mono_verify.debug.tsv
BUILD_BIN = build/mono27b_chat
REF_DEBUG = /mnt/mydata/projects2/mono27b/ref/llama.cpp/build/bin/llama-debug

.PHONY: verify verify-only compare-all e2e build gen-our gen-ref run-checks clean

# =========================================================================
# FULL VERIFY: build + generate BOTH ref + our data + all comparisons
# =========================================================================
verify: build gen-our gen-ref run-checks compare-all

verify-only: run-checks compare-all

# =========================================================================
# BUILD
# =========================================================================
build:
	@echo ""
	@echo "=== Building mono27b_chat ==="
	@cmake --build build -j$$(nproc) 2>/dev/null || \
	  (cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
	   -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release && \
	   cmake --build build -j$$(nproc))

# =========================================================================
# GENERATE OUR DATA
# =========================================================================
gen-our:
	@echo ""
	@echo "=== Generating OUR debug data ==="
	@mkdir -p debug/verify
	@rm -f $(DEBUG)
	$(BUILD_BIN) -m $(GGUF) -p "$(PROMPT)" --gen 1 --ctx 4096 --seed 944990222 \
		--trace /dev/null --debug $(DEBUG) > /dev/null 2>&1
	@cd debug/verify && python3 extract_data.py $(DEBUG)
	@# Save our logits as binary (prompt token = pos 0)
	@cp /tmp/our_logits_0.bin debug/verify/our_logits.bin 2>/dev/null || true
	@echo "  Our data ready."

# =========================================================================
# GENERATE REFERENCE DATA (llama.cpp)
# =========================================================================
gen-ref:
	@echo ""
	@echo "=== Generating REFERENCE data (llama-debug) ==="
	@# Save reference logits
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--save-logits --logits-output-dir /tmp/ref_logits 2>/dev/null
	@cp /tmp/ref_logits/llamacpp-Qwen3.6-27B-UD-Q4_K_XL.bin \
	   debug/verify/ref_logits.bin 2>/dev/null || true
	@# Save reference intermediate tensors (SSM layer 0 + attention layer 3)
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
	--tensor-filter 'model.input_embed$$|attn_norm-0$$|linear_attn_qkv_mixed-0$$|z-0$$|conv_output_raw-0$$|conv_output_silu-0$$|q_conv_predelta-0$$|k_conv_predelta-0$$|attn_output-0$$|final_output-0$$|linear_attn_out-0$$|post_attention_norm-0$$|ffn_gate-0$$|ffn_up-0$$|ffn_mul-0$$|ffn_down-0$$|l_out-0$$|attn_norm-3$$|Qcur_full-3$$|Qcur_normed-3$$|attn_pregate-3$$|attn_gated-3$$|attn_output-3$$|attn_post_norm-3$$|ffn_gate-3$$|ffn_up-3$$|ffn_out-3$$|l_out-3$$' \
		> /tmp/ref_stdout.txt 2>/tmp/ref_stderr.txt
	@cp /tmp/ref_stdout.txt debug/verify/ref_intermediates.txt 2>/dev/null || true
	@echo "  Reference data ready."

# =========================================================================
# COMPONENT CHECKS (unit tests: Q5_K, Q6_K, RMS norm, M-RoPE, etc.)
# =========================================================================
run-checks:
	@echo ""
	@echo "=== Running Component Checks ==="
	@cd debug/verify && python3 run_all_checks.py \
		--gguf $(GGUF) --debug $(DEBUG) --ref-intermediates ref_intermediates.txt

# =========================================================================
# COMPREHENSIVE TENSOR COMPARISON (all comparable tensors, all layers)
# =========================================================================
compare-all:
	@echo ""
	@echo "=== Running Comprehensive Tensor Comparison ==="
	@cd debug/verify && python3 compare_all.py \
		--our-debug $(DEBUG) \
		--ref ref_intermediates.txt \
		--ref-logits ref_logits.bin \
		--our-logits our_logits.bin

# =========================================================================
# END-TO-END LOGIT COMPARISON (standalone)
# =========================================================================
e2e: build gen-our gen-ref
	@cd debug/verify && python3 compare_e2e.py

# =========================================================================
# CLEAN
# =========================================================================
clean:
	rm -f debug/verify/embed_full.txt debug/verify/attn_norm_full.txt
	rm -f debug/verify/ref_intermediates.txt debug/verify/ref_logits.bin debug/verify/our_logits.bin
