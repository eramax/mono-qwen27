#include "mono27b_cli.h"
#include "mono27b_format.h"
#include "mono27b_packer.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    Mono27BPackArgs args;
    if (!mono27b_parse_pack_args(argc, argv, args) || args.show_help) {
        mono27b_print_pack_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }
    if (!args.blob_check_path.empty()) {
        std::string error;
        if (!mono27b_validate_blob_file(args.blob_check_path, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        std::fprintf(stderr, "blob ok: %s\n", args.blob_check_path.c_str());
        return 0;
    }

    std::string status;
    std::string error;
    if (!mono27b_pack_models(args.target_gguf, args.draft_gguf, args.out_blob, status, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "%s\n", status.c_str());
    return 0;
}
