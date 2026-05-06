.PHONY: build test quick-test large-test clean verify verify-all e2e e2e-text generate-verify-data

MODEL_PATH ?= /mnt/mydata/projects2/mono27b/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
PROMPT ?= "give me 2 py example"
GEN ?= 500
CTX ?= 8192
SEED ?= 944990222

build:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DMONO27B_TIMING=OFF
	cmake --build build -j$(shell nproc)

test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p $(PROMPT) --gen $(GEN) --ctx $(CTX) --seed $(SEED)

run: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "$(PROMPT)" --chat --ctx 8192 --gen 8192

quick-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "The quick brown fox" --gen 5 --ctx 2048

large-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "Artificial intelligence is" --gen 20 --ctx 8192

verify: build
	$(MAKE) -f debug/verify/Makefile verify GGUF=$(MODEL_PATH)

verify-only:
	$(MAKE) -f debug/verify/Makefile verify-only GGUF=$(MODEL_PATH)

e2e: build
	$(MAKE) -f debug/verify/Makefile e2e GGUF=$(MODEL_PATH)

e2e-text: build
	$(MAKE) -f debug/verify/Makefile e2e-text GGUF=$(MODEL_PATH)

e2e-text-all: build
	$(MAKE) -f debug/verify/Makefile e2e-text-all GGUF=$(MODEL_PATH)

clean:
	rm -rf build

compare-perf:
	@echo "Building timing-enabled binary..."
	@rm -f build/CMakeCache.txt
	@cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DMONO27B_TIMING=ON
	@cmake --build build -j$(shell nproc)
	@echo "Running mono27b with timing instrumentation (gen 30)..."
	@./build/mono27b_chat -m $(MODEL_PATH) -p "The quick brown fox" --gen 30 --ctx $(CTX) --seed $(SEED) > /tmp/mono27b_timing.txt 2>&1
	@echo "Running llama-bench for reference..."
	@/mnt/data1/projects/llm/llama.cpp/build/bin/llama-bench -m $(MODEL_PATH) -p 4 -n 30 > /tmp/llama_bench.txt 2>&1
	@python3 scripts/compare_perf.py /tmp/mono27b_timing.txt $(MODEL_PATH)

build-timing:
	@rm -f build/CMakeCache.txt
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DMONO27B_TIMING=ON
	cmake --build build -j$(shell nproc)

build-fast:
	@rm -f build/CMakeCache.txt
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DMONO27B_TIMING=OFF
	cmake --build build -j$(shell nproc)

help:
	@echo "Usage: make [target] [VAR=value]"
	@echo ""
	@echo "Targets:"
	@echo "  build        - Configure and build mono27b_chat"
	@echo "  build-fast   - Build without timing instrumentation"
	@echo "  build-timing - Build with per-step CUDA timing"
	@echo "  test         - Run chat with custom parameters"
	@echo "  run          - Run with a prompt (ctx=128k, default sampling)"
	@echo "  quick-test   - Run quick 5-token generation"
	@echo "  large-test   - Run 20-token generation"
	@echo "  compare-perf - Run timing comparison vs llama.cpp"
	@echo "  verify       - Run all verification scripts (generates data first)"
	@echo "  verify-all   - Rebuild, generate data, run all verifications (incl. text)"
	@echo "  e2e          - End-to-end logit comparison (single token)"
	@echo "  e2e-text     - End-to-end text comparison (50 tokens, --chat mode)"
	@echo "  clean        - Remove build directory"
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
	@echo "  make run PROMPT='Hello world'"
	@echo "  make run PROMPT='Explain quantum computing' GEN=200"
	@echo "  make test PROMPT='Hello world' GEN=15 CTX=2048"
	@echo "  make quick-test"
	@echo "  make compare-perf"
	@echo "  make verify"
