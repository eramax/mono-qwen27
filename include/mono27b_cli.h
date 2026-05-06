#pragma once

#include <cstdint>
#include <string>

struct Mono27BChatArgs {
    std::string model;
    std::string prompt;
    uint32_t seed  = 944990222U;
    int max_ctx    = 131072;
    int max_gen    = 512;
    bool chat      = false;
    bool verbose   = false;
    bool quiet     = false;
    bool show_help = false;
};

bool mono27b_parse_args(int argc, char ** argv, Mono27BChatArgs & out);
void mono27b_print_usage(const char * prog);

struct Mono27BPackArgs {
    std::string target_gguf;
    std::string draft_gguf;
    std::string out_blob;
    std::string blob_check_path;
    bool show_help = false;
};

bool mono27b_parse_pack_args(int argc, char ** argv, Mono27BPackArgs & out);
void mono27b_print_pack_usage(const char * prog);
