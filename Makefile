.PHONY: build test clean

BLOB_PATH ?= test.m27b
PROMPT ?= "What is"
GEN ?= 10
CTX ?= 4096

build:
	cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(shell nproc)

test: build
	./build/mono27b_chat --blob $(BLOB_PATH) -p "$(PROMPT)" --gen $(GEN) --ctx $(CTX)

quick-test: build
	./build/mono27b_chat --blob $(BLOB_PATH) -p "The quick brown fox" --gen 5 --ctx 2048

large-test: build
	./build/mono27b_chat --blob $(BLOB_PATH) -p "Artificial intelligence is" --gen 20 --ctx 8192

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
	@echo "  BLOB_PATH   - Path to .m27b blob file (default: test.m27b)"
	@echo "  PROMPT      - Input prompt string (default: 'What is')"
	@echo "  GEN         - Number of tokens to generate (default: 10)"
	@echo "  CTX         - Context size in tokens (default: 4096)"
	@echo ""
	@echo "Examples:"
	@echo "  make test PROMPT='Hello world' GEN=15 CTX=2048"
	@echo "  make quick-test"
	@echo "  make large-test"
