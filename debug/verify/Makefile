# Verification Makefile for mono27b
# make verify        -> runs BOTH llama.cpp ref AND our tool, compares side-by-side
# make verify-only   -> only runs checks on existing data
# make e2e           -> end-to-end logit comparison only

GGUF ?= /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
DEBUG ?= /tmp/mono_verify.debug.tsv
BUILD_BIN = build/mono27b_chat
REF_DEBUG = /mnt/mydata/projects2/mono27b/ref/llama.cpp/build/bin/llama-debug

.PHONY: verify verify-only e2e generate-data clean

# =========================================================================
# FULL VERIFY: build + generate BOTH ref + our data + compare + summary
# =========================================================================
verify: build gen-our gen-ref run-checks

verify-only: run-checks

build:
	@echo ""
	@echo "=== Building mono27b_chat ==="
	@cmake --build build -j$$(nproc) 2>/dev/null || \
	  (cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
	   -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release && \
	   cmake --build build -j$$(nproc))

# --- Generate our data ---
gen-our:
	@echo ""
	@echo "=== Generating OUR debug data ==="
	@mkdir -p debug/verify
	@rm -f $(DEBUG)
	$(BUILD_BIN) -m $(GGUF) -p "give" --gen 1 --ctx 4096 --seed 944990222 \
		--trace /dev/null --debug $(DEBUG) > /dev/null 2>&1
	@cd debug/verify && python3 extract_data.py $(DEBUG)
	@# Save our logits as binary
	@cp /tmp/mono_logits.bin debug/verify/our_logits.bin 2>/dev/null || true
	@echo "  Our data ready."

# --- Generate reference data ---
gen-ref:
	@echo ""
	@echo "=== Generating REFERENCE data (llama-debug) ==="
	@# Save reference logits
	$(REF_DEBUG) -m $(GGUF) -p "give" -c 4096 -b 1 -ub 1 \
		--save-logits --logits-output-dir /tmp/ref_logits 2>/dev/null
	@cp /tmp/ref_logits/llamacpp-Qwen3.6-27B-UD-Q4_K_XL.bin \
	   debug/verify/ref_logits.bin 2>/dev/null || true
	@# Save reference intermediate tensors (SSM layer 0)
	$(REF_DEBUG) -m $(GGUF) -p "give" -c 4096 -b 1 -ub 1 \
	--tensor-filter "attn_norm-0|conv_output_silu-0|q_conv_predelta-0|k_conv_predelta-0|attn_output-0|final_output-0|linear_attn_out-0|post_attention_norm-0|ffn_gate-0|ffn_up-0|ffn_mul-0|ffn_down-0|l_out-0|model.input_embed" \
		> /tmp/ref_stdout.txt 2>/tmp/ref_stderr.txt
	@cp /tmp/ref_stdout.txt debug/verify/ref_intermediates.txt 2>/dev/null || true
	@echo "  Reference data ready."

# =========================================================================
# RUN ALL CHECKS (uses both ref and our data)
# =========================================================================
run-checks:
	@echo ""
	@echo "=== Running All Verifications ==="
	@cd debug/verify && python3 run_all_checks.py \
		--gguf $(GGUF) --debug $(DEBUG) --ref-intermediates ref_intermediates.txt

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
	rm -f debug/verify/ref_intermediates.txt
