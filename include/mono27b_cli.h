#pragma once

#include <string>

struct Mono27BChatArgs {
    std::string blob_path;
    std::string target_gguf;
    std::string draft_gguf;
    std::string prompt;
    std::string trace_path;
    std::string debug_path;
    int max_ctx = 131072;
    int max_gen = 512;
    bool show_help = false;
};

struct Mono27BPackArgs {
    std::string target_gguf;
    std::string draft_gguf;
    std::string out_blob;
    std::string blob_check_path;
    bool show_help = false;
};

bool mono27b_parse_chat_args(int argc, char ** argv, Mono27BChatArgs & out);
bool mono27b_parse_pack_args(int argc, char ** argv, Mono27BPackArgs & out);
void mono27b_print_chat_usage(const char * prog);
void mono27b_print_pack_usage(const char * prog);
