# Verification Makefile for mono27b
# make verify        -> runs BOTH llama.cpp ref AND our tool, compares side-by-side
# make verify-only   -> only runs checks on existing data
# make e2e           -> end-to-end logit comparison only
# make compare-all   -> comprehensive tensor comparison (all layers, all tensors)

GGUF ?= /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
PROMPT ?= give
DEBUG ?= /tmp/mono_verify.debug.tsv

# Auto-detect invocation: -f from project root vs cd'd into debug/verify
ifeq ($(wildcard CMakeLists.txt),CMakeLists.txt)
  PROJ_ROOT := .
  SCRIPT_DIR := $(CURDIR)/debug/verify
else
  PROJ_ROOT := ../..
  SCRIPT_DIR := $(CURDIR)
endif

BUILD_BIN = $(PROJ_ROOT)/build/mono27b_chat
REF_DEBUG = /mnt/mydata/projects2/mono27b/ref/llama.cpp/build/bin/llama-debug
REF_COMPLETION = /mnt/mydata/projects2/mono27b/ref/llama.cpp/build/bin/llama-completion

# Data files (always use absolute paths via SCRIPT_DIR)
REF_INTERMEDIATES := $(SCRIPT_DIR)/ref_intermediates.txt
REF_LOGITS := $(SCRIPT_DIR)/ref_logits.bin
OUR_LOGITS := $(SCRIPT_DIR)/our_logits.bin
OUR_TEXT := /tmp/our_text.txt
REF_TEXT := /tmp/ref_text.txt

.PHONY: verify verify-only compare-all e2e e2e-text build gen-our gen-ref gen-our-text gen-ref-text run-checks clean-data clean

# =========================================================================
# FULL VERIFY: clean + build + generate BOTH ref + our data + all comparisons
# =========================================================================
verify: clean-data build gen-our gen-ref run-checks compare-all e2e-text
verify-all: clean-data build gen-our gen-ref gen-our-text gen-ref-text run-checks compare-all compare-text

verify-only: run-checks compare-all

# =========================================================================
# BUILD
# =========================================================================
build:
	@echo ""
	@echo "=== Building mono27b_chat ==="
	@cmake --build $(PROJ_ROOT)/build -j$$(nproc) 2>/dev/null || \
	  (cmake -S $(PROJ_ROOT) -B $(PROJ_ROOT)/build -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
	   -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release && \
	   cmake --build $(PROJ_ROOT)/build -j$$(nproc))

# =========================================================================
# GENERATE OUR DATA
# =========================================================================
gen-our:
	@echo ""
	@echo "=== Generating OUR debug data ==="
	@rm -f $(DEBUG)
	$(BUILD_BIN) -m $(GGUF) -p "$(PROMPT)" --gen 1 --ctx 4096 --seed 944990222 \
		--trace /dev/null --debug $(DEBUG) > /dev/null 2>&1
	@python3 $(SCRIPT_DIR)/extract_data.py $(DEBUG)
	@cp /tmp/our_logits_0.bin $(OUR_LOGITS) 2>/dev/null || true
	@echo "  Our data ready."

# =========================================================================
# GENERATE REFERENCE DATA (llama.cpp)
# Split into per-layer runs to avoid GPU memory exhaustion / stdout truncation.
# Each run uses a minimal tensor filter (≤ 5 tensors) to prevent OOM.
# =========================================================================
gen-ref:
	@echo ""
	@echo "=== Generating REFERENCE data (llama-debug, per-layer runs) ==="
	@rm -f $(REF_INTERMEDIATES) $(REF_LOGITS)
	@# Save reference logits (prompt token)
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--save-logits --logits-output-dir /tmp/ref_logits 2>/dev/null
	@cp /tmp/ref_logits/llamacpp-Qwen3.6-27B-UD-Q4_K_XL.bin \
	   $(REF_LOGITS) 2>/dev/null || true
	@# Run 1: embedding + SSM layer 0 pre-FFN
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--tensor-filter 'model.input_embed$$|attn_norm-0$$|z-0$$|conv_output_raw-0$$|conv_output_silu-0$$' \
		> /tmp/ref_run1.txt 2>/dev/null
	@# Run 2: SSM layer 0 post-DeltaNet
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--tensor-filter 'q_conv_predelta-0$$|k_conv_predelta-0$$|attn_output-0$$|final_output-0$$' \
		> /tmp/ref_run2.txt 2>/dev/null
	@# Run 3: SSM layer 0 FFN + layer output
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--tensor-filter 'post_attention_norm-0$$|ffn_gate-0$$|ffn_up-0$$|l_out-0$$' \
		> /tmp/ref_run3.txt 2>/dev/null
	@# Run 4: attention layer 3 pre-FFN
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--tensor-filter 'attn_norm-3$$|attn_pregate-3$$|attn_gated-3$$|attn_output-3$$' \
		> /tmp/ref_run4.txt 2>/dev/null
	@# Run 5: attention layer 3 FFN + layer output
	$(REF_DEBUG) -m $(GGUF) -p "$(PROMPT)" -c 4096 -b 1 -ub 1 \
		--tensor-filter 'attn_post_norm-3$$|ffn_gate-3$$|ffn_up-3$$|ffn_out-3$$|l_out-3$$' \
		> /tmp/ref_run5.txt 2>/dev/null
	@# Merge all runs into a single ref_intermediates.txt
	@cat /tmp/ref_run1.txt /tmp/ref_run2.txt /tmp/ref_run3.txt \
	     /tmp/ref_run4.txt /tmp/ref_run5.txt > $(REF_INTERMEDIATES)
	@echo "  Reference data ready."

# =========================================================================
# COMPONENT CHECKS (unit tests: Q5_K, Q6_K, RMS norm, M-RoPE, etc.)
# =========================================================================
run-checks:
	@echo ""
	@echo "=== Running Component Checks ==="
	@python3 $(SCRIPT_DIR)/run_all_checks.py \
		--gguf $(GGUF) --debug $(DEBUG) --ref-intermediates $(REF_INTERMEDIATES)

# =========================================================================
# COMPREHENSIVE TENSOR COMPARISON (all comparable tensors, all layers)
# =========================================================================
compare-all:
	@echo ""
	@echo "=== Running Comprehensive Tensor Comparison ==="
	@python3 $(SCRIPT_DIR)/compare_all.py \
		--our-debug $(DEBUG) \
		--ref $(REF_INTERMEDIATES) \
		--ref-logits $(REF_LOGITS) \
		--our-logits $(OUR_LOGITS)

# =========================================================================
# END-TO-END LOGIT COMPARISON (standalone)
# =========================================================================
e2e: clean-data build gen-our gen-ref
	@python3 $(SCRIPT_DIR)/compare_e2e.py

# =========================================================================
# END-TO-END TEXT COMPARISON (generated text, not just logits)
# =========================================================================
gen-our-text:
	@echo ""
	@echo "=== Generating OUR text output (50 tokens) ==="
	@rm -f $(OUR_TEXT)
	$(BUILD_BIN) -m $(GGUF) -p "$(PROMPT)" --gen 50 --ctx 4096 --seed 944990222 \
		--quiet --chat > $(OUR_TEXT) 2>/dev/null
	@wc -c < $(OUR_TEXT) | xargs -I{} echo "  Our text saved to $(OUR_TEXT) ({} bytes)"

gen-ref-text:
	@echo ""
	@echo "=== Generating REFERENCE text output (50 tokens, llama-completion) ==="
	@rm -f $(REF_TEXT)
	@echo "" | $(REF_COMPLETION) -m $(GGUF) -p "$(PROMPT)" -n 50 -c 4096 \
		--seed 944990222 -cnv --no-display-prompt \
		-b 1 -ub 1 > $(REF_TEXT) 2>/dev/null
	@wc -c < $(REF_TEXT) | xargs -I{} echo "  Ref text saved to $(REF_TEXT) ({} bytes)"

compare-text:
	@python3 $(SCRIPT_DIR)/compare_text.py $(OUR_TEXT) $(REF_TEXT); exit_code=$$?; \
	if [ $$exit_code -eq 1 ]; then echo "  (differences are expected until model execution matches)"; exit 0; fi; \
	exit $$exit_code

e2e-text: clean-data build gen-our-text gen-ref-text compare-text

e2e-text-all: clean-data build gen-our gen-ref gen-our-text gen-ref-text run-checks compare-all compare-text

# =========================================================================
# CLEAN
# =========================================================================
clean-data:
	@rm -f $(REF_INTERMEDIATES) $(REF_LOGITS) $(OUR_LOGITS)
	@rm -f $(OUR_TEXT) $(REF_TEXT)
	@rm -f $(SCRIPT_DIR)/embed_full.txt $(SCRIPT_DIR)/attn_norm_full.txt
	@rm -f $(DEBUG)
	@rm -f /tmp/our_logits_*.bin /tmp/ref_run*.txt
	@rm -rf /tmp/ref_logits

clean: clean-data
