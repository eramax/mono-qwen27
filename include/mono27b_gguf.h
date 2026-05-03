#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t MONO27B_GGUF_VERSION = 3U;
constexpr uint32_t MONO27B_GGUF_DEFAULT_ALIGNMENT = 32U;

enum Mono27BGgufType : uint32_t {
    MONO27B_GGUF_TYPE_UINT8 = 0,
    MONO27B_GGUF_TYPE_INT8 = 1,
    MONO27B_GGUF_TYPE_UINT16 = 2,
    MONO27B_GGUF_TYPE_INT16 = 3,
    MONO27B_GGUF_TYPE_UINT32 = 4,
    MONO27B_GGUF_TYPE_INT32 = 5,
    MONO27B_GGUF_TYPE_FLOAT32 = 6,
    MONO27B_GGUF_TYPE_BOOL = 7,
    MONO27B_GGUF_TYPE_STRING = 8,
    MONO27B_GGUF_TYPE_ARRAY = 9,
    MONO27B_GGUF_TYPE_UINT64 = 10,
    MONO27B_GGUF_TYPE_INT64 = 11,
    MONO27B_GGUF_TYPE_FLOAT64 = 12,
};

enum Mono27BGgmlType : uint32_t {
    MONO27B_GGML_TYPE_F32 = 0,
    MONO27B_GGML_TYPE_F16 = 1,
    MONO27B_GGML_TYPE_Q4_0 = 2,
    MONO27B_GGML_TYPE_Q4_1 = 3,
    MONO27B_GGML_TYPE_Q5_0 = 6,
    MONO27B_GGML_TYPE_Q5_1 = 7,
    MONO27B_GGML_TYPE_Q8_0 = 8,
    MONO27B_GGML_TYPE_Q8_1 = 9,
    MONO27B_GGML_TYPE_Q2_K = 10,
    MONO27B_GGML_TYPE_Q3_K = 11,
    MONO27B_GGML_TYPE_Q4_K = 12,
    MONO27B_GGML_TYPE_Q5_K = 13,
    MONO27B_GGML_TYPE_Q6_K = 14,
    MONO27B_GGML_TYPE_Q8_K = 15,
};

struct Mono27BGgufTensorInfo {
    std::string name;
    uint32_t ggml_type = 0;
    uint64_t offset = 0;
    uint64_t size_bytes = 0;
};

struct Mono27BGgufMetadata {
    std::string architecture;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    uint32_t vocab_size = 0;
    uint32_t merges_count = 0;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    uint32_t im_start_id = 0;
    uint32_t im_end_id = 0;
    uint32_t embedding_length = 0;
    uint32_t block_count = 0;
    uint32_t feed_forward_length = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
};

struct Mono27BGgufFile {
    uint32_t version = 0;
    uint32_t alignment = MONO27B_GGUF_DEFAULT_ALIGNMENT;
    uint64_t data_offset = 0;
    Mono27BGgufMetadata metadata;
    std::vector<Mono27BGgufTensorInfo> tensors;
};

bool mono27b_read_gguf(const std::string & path, Mono27BGgufFile & out, std::string & error);
size_t mono27b_ggml_row_size(uint32_t ggml_type, uint32_t ne);
