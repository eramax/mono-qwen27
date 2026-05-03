#pragma once

#include "mono27b_format.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Mono27BTokenizer {
public:
    bool load_from_blob(const std::string & blob_path,
                        const Mono27BBlobHeader & header,
                        std::string & error);

    uint32_t bos_id() const { return tokenizer_.bos_id; }
    uint32_t eos_id() const { return tokenizer_.eos_id; }
    uint32_t vocab_size() const { return tokenizer_.vocab_size; }
    uint32_t merges_count() const { return tokenizer_.merges_count; }
    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<int32_t> bpe(const std::string & word) const;
    std::string decode_one(int32_t id) const;
    std::string decode(const std::vector<int32_t> & ids) const;
    bool is_terminal(int32_t id) const;
    bool is_eos(int32_t id) const { return is_terminal(id); }

private:
    void init_byte_tables();
    bool should_skip_token(int32_t id) const;
    static size_t utf8_len(char c);

    Mono27BTokenizerSection tokenizer_{};
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> str_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
    std::array<std::string, 256> byte_to_unicode_{};
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
};
