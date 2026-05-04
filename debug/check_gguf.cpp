#include "include/mono27b_gguf.h"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) return 1;
    Mono27BGgufFile gguf;
    std::string error;
    if (!mono27b_read_gguf(argv[1], gguf, error)) {
        std::printf("Error: %s\n", error.c_str());
        return 1;
    }
    
    int start = std::atoi(argv[2]);
    int count = (argc > 3) ? std::atoi(argv[3]) : 1;
    for (int i = start; i < start + count && i < (int)gguf.metadata.tokens.size(); ++i) {
        std::printf("Token %d: '%s'\n", i, gguf.metadata.tokens[i].c_str());
    }
    
    return 0;
}
