.PHONY: build test quick-test large-test clean verify verify-all generate-verify-data

MODEL_PATH ?= /mnt/mydata/projects2/mono27b/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
PROMPT ?= "give me 2 py example"
GEN ?= 500
CTX ?= 4096
SEED ?= 944990222

build:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(shell nproc)

test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p $(PROMPT) --gen $(GEN) --ctx $(CTX) --seed $(SEED) --quiet

quick-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "The quick brown fox" --gen 5 --ctx 2048 --quiet

large-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "Artificial intelligence is" --gen 20 --ctx 8192 --quiet

verify: build
	$(MAKE) -f debug/verify/Makefile verify GGUF=$(MODEL_PATH)

verify-only:
	$(MAKE) -f debug/verify/Makefile verify-only GGUF=$(MODEL_PATH)

e2e: build
	$(MAKE) -f debug/verify/Makefile e2e GGUF=$(MODEL_PATH)

clean:
	rm -rf build

help:
	@echo "Usage: make [target] [VAR=value]"
	@echo ""
	@echo "Targets:"
	@echo "  build       - Configure and build mono27b_chat"
	@echo "  test        - Run chat with custom parameters"
	@echo "  quick-test  - Run quick 5-token generation"
	@echo "  large-test  - Run 20-token generation"
	@echo "  verify      - Run all verification scripts (generates data first)"
	@echo "  verify-all  - Rebuild, generate data, run all verifications"
	@echo "  clean       - Remove build directory"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_PATH  - Path to target GGUF model"
	@echo "  PROMPT      - Input prompt string"
	@echo "  GEN         - Number of tokens to generate"
	@echo "  CTX         - Context size in tokens"
	@echo ""
	@echo "Verification scripts in debug/verify/:"
	@echo "  verify_rms_norm.py    - RMS norm (full 5120-element comparison)"
	@echo "  test_q5k_matvec.py    - Q5_K matvec (wqkv_gate) vs GPU"
	@echo "  verify_q6k_full.py    - Q6_K matvec (wqkv) vs GPU (fixed dequant)"
	@echo "  verify_deltanet.py    - Python DeltaNet reference impl"
	@echo ""
	@echo "Examples:"
	@echo "  make test PROMPT='Hello world' GEN=15 CTX=2048"
	@echo "  make quick-test"
	@echo "  make verify"
