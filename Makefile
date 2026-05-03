.PHONY: build test clean

MODEL_PATH ?= /mnt/mydata/projects2/specfusion/model/Qwen3.6-27B-UD-Q4_K_XL.gguf
PROMPT ?= "give me 2 py example"
GEN ?= 10
CTX ?= 4096

build:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(shell nproc)

test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p $(PROMPT) --gen $(GEN) --ctx $(CTX)

quick-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "The quick brown fox" --gen 5 --ctx 2048

large-test: build
	./build/mono27b_chat -m $(MODEL_PATH) -p "Artificial intelligence is" --gen 20 --ctx 8192

clean:
	rm -rf build

help:
	@echo "Usage: make [target] [VAR=value]"
	@echo ""
	@echo "Targets:"
	@echo "  build       - Configure and build mono27b_chat"
	@echo "  test        - Run chat with custom parameters (default: 10 tokens, ctx=4096)"
	@echo "  quick-test  - Run quick 5-token generation (ctx=2048)"
	@echo "  large-test  - Run 20-token generation (ctx=8192)"
	@echo "  clean       - Remove build directory"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_PATH  - Path to target GGUF model"
	@echo "  PROMPT      - Input prompt string (default: 'give me 2 py example')"
	@echo "  GEN         - Number of tokens to generate (default: 10)"
	@echo "  CTX         - Context size in tokens (default: 4096)"
	@echo ""
	@echo "Examples:"
	@echo "  make test PROMPT='Hello world' GEN=15 CTX=2048"
	@echo "  make quick-test"
	@echo "  make large-test"
