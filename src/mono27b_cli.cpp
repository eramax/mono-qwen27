#include "mono27b_cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>

void mono27b_print_chat_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --blob model.m27b --prompt \"...\" [--ctx N] [--gen N] [--trace PATH]\n"
        "       %s --blob model.m27b -p \"...\" [--ctx N] [--gen N] [--trace PATH]\n",
        prog,
        prog);
    std::fprintf(stderr,
        "       %s -m target.gguf -p \"...\" [--ctx N] [--gen N] [--trace PATH]\n",
        prog);
}

void mono27b_print_pack_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --target target.gguf --draft draft.gguf --out model.m27b\n"
        "       %s --blob-only-check model.m27b\n",
        prog,
        prog);
}

bool mono27b_parse_chat_args(int argc, char ** argv, Mono27BChatArgs & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            out.show_help = true;
            return true;
        }
        if (arg == "--blob" && i + 1 < argc) {
            out.blob_path = argv[++i];
            continue;
        }
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            out.target_gguf = argv[++i];
            continue;
        }
        if ((arg == "-md" || arg == "--model-draft") && i + 1 < argc) {
            out.draft_gguf = argv[++i];
            continue;
        }
        if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            out.prompt = argv[++i];
            continue;
        }
        if (arg == "--ctx" && i + 1 < argc) {
            out.max_ctx = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--gen" && i + 1 < argc) {
            out.max_gen = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--trace" && i + 1 < argc) {
            out.trace_path = argv[++i];
            continue;
        }
        return false;
    }
    const bool has_blob = !out.blob_path.empty();
    const bool has_models = !out.target_gguf.empty();
    return !out.prompt.empty() && (has_blob || has_models);
}

bool mono27b_parse_pack_args(int argc, char ** argv, Mono27BPackArgs & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            out.show_help = true;
            return true;
        }
        if (arg == "--target" && i + 1 < argc) {
            out.target_gguf = argv[++i];
            continue;
        }
        if (arg == "--draft" && i + 1 < argc) {
            out.draft_gguf = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out.out_blob = argv[++i];
            continue;
        }
        if (arg == "--blob-only-check" && i + 1 < argc) {
            out.blob_check_path = argv[++i];
            continue;
        }
        return false;
    }
    if (!out.blob_check_path.empty()) {
        return true;
    }
    return !out.target_gguf.empty() && !out.draft_gguf.empty() && !out.out_blob.empty();
}
