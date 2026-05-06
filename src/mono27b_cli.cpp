#include "mono27b_cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>

void mono27b_print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf -p \"prompt\" [options]\n"
        "options:\n"
        "  -m, --model PATH   model GGUF file\n"
        "  -p, --prompt TEXT  input prompt\n"
        "  --ctx N            context size (default: 131072)\n"
        "  --gen N            tokens to generate (default: 512)\n"
        "  --seed N           rng seed (default: 944990222)\n"
        "  --chat             apply chat template\n"
        "  --trace-gen        print per-token generation timing to stderr\n"
        "  --quiet            suppress stats output\n"
        "  -v, --verbose      verbose logging\n"
        "  -h, --help         show this help\n",
        prog);
}

bool mono27b_parse_args(int argc, char **argv, Mono27BChatArgs &out) {
    auto match = [](const std::string &arg, const char *opt1, const char *opt2) {
        return arg == opt1 || (opt2 && arg == opt2);
    };
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (match(arg, "-h", "--help")) { out.show_help = true; return true; }
        if (match(arg, "-m", "--model") && i + 1 < argc) { out.model = argv[++i]; continue; }
        if (match(arg, "-p", "--prompt") && i + 1 < argc) { out.prompt = argv[++i]; continue; }
        if (arg == "--ctx" && i + 1 < argc) { out.max_ctx = std::atoi(argv[++i]); continue; }
        if (arg == "--gen" && i + 1 < argc) { out.max_gen = std::atoi(argv[++i]); continue; }
        if (arg == "--seed" && i + 1 < argc) { out.seed = std::strtoul(argv[++i], nullptr, 10); continue; }
        if (arg == "--chat") { out.chat = true; continue; }
        if (arg == "--trace-gen") { out.trace_gen = true; continue; }
        if (arg == "--quiet") { out.quiet = true; continue; }
        if (match(arg, "-v", "--verbose")) { out.verbose = true; continue; }
        std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        return false;
    }
    return !out.model.empty() && !out.prompt.empty();
}

void mono27b_print_pack_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --target target.gguf --draft draft.gguf --out model.m27b\n"
        "       %s --blob-only-check model.m27b\n",
        prog, prog);
}

bool mono27b_parse_pack_args(int argc, char ** argv, Mono27BPackArgs & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { out.show_help = true; return true; }
        if (arg == "--target" && i + 1 < argc) { out.target_gguf = argv[++i]; continue; }
        if (arg == "--draft" && i + 1 < argc) { out.draft_gguf = argv[++i]; continue; }
        if (arg == "--out" && i + 1 < argc) { out.out_blob = argv[++i]; continue; }
        if (arg == "--blob-only-check" && i + 1 < argc) { out.blob_check_path = argv[++i]; continue; }
        return false;
    }
    return !out.blob_check_path.empty() ||
           (!out.target_gguf.empty() && !out.draft_gguf.empty() && !out.out_blob.empty());
}
