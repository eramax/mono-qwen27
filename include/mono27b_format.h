#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

constexpr uint32_t MONO27B_BLOB_MAGIC = 0x4237324dU;
constexpr uint32_t MONO27B_SCHEMA_VERSION = 4U;
constexpr uint32_t MONO27B_TARGET_LAYERS = 64U;
constexpr uint32_t MONO27B_DRAFT_LAYERS = 5U;
constexpr uint32_t MONO27B_TARGET_HIDDEN = 5120U;
constexpr uint32_t MONO27B_TARGET_FFN = 17408U;
constexpr uint32_t MONO27B_TARGET_HEADS = 24U;
constexpr uint32_t MONO27B_TARGET_KV_HEADS = 4U;

constexpr char MONO27B_TARGET_ARCH[] = "qwen35";
constexpr char MONO27B_DRAFT_ARCH[] = "dflash-draft";
constexpr char MONO27B_TARGET_QUANT[] = "q4_k_m";
constexpr char MONO27B_DRAFT_QUANT[] = "q8_0";
constexpr uint32_t MONO27B_SECTION_TOKENIZER = 1U;
constexpr uint32_t MONO27B_SECTION_WEIGHTS = 2U;
constexpr uint32_t MONO27B_TOKENIZER_MAGIC = 0x544b324dU;
constexpr uint32_t MONO27B_WEIGHTS_MAGIC = 0x5754324dU;
constexpr uint32_t MONO27B_SOURCE_TARGET = 1U;
constexpr uint32_t MONO27B_SOURCE_DRAFT = 2U;
constexpr uint32_t MONO27B_ROLE_TARGET_MATVEC = 1U;
constexpr uint32_t MONO27B_ROLE_DRAFT_MATVEC = 2U;
constexpr uint32_t MONO27B_ROLE_TARGET_OUTPUT_HEAD = 3U;

struct Mono27BBlobHeader {
    uint32_t magic = MONO27B_BLOB_MAGIC;
    uint32_t schema_version = MONO27B_SCHEMA_VERSION;
    uint32_t section_count = 0;
    uint32_t reserved0 = 0;
    uint64_t file_bytes = 0;
    uint64_t tensor_table_offset = 0;
    uint64_t section_table_offset = 0;
    uint64_t tokenizer_offset = 0;
    uint64_t tokenizer_bytes = 0;
    uint32_t max_ctx_hint = 8192;
    uint32_t target_layers = MONO27B_TARGET_LAYERS;
    uint32_t draft_layers = MONO27B_DRAFT_LAYERS;
    char target_arch[16] = {};
    char draft_arch[16] = {};
    char target_quant[16] = {};
    char draft_quant[16] = {};
};

struct Mono27BBlobSection {
    uint32_t kind = 0;
    uint32_t reserved = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct Mono27BTokenizerSection {
    uint32_t magic = MONO27B_TOKENIZER_MAGIC;
    uint32_t version = 1U;
    uint32_t vocab_size = 0;
    uint32_t merges_count = 0;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    uint32_t im_start_id = 0;
    uint32_t im_end_id = 0;
};

struct Mono27BWeightsSection {
    uint32_t magic = MONO27B_WEIGHTS_MAGIC;
    uint32_t version = 1U;
    uint32_t entry_count = 0;
    uint32_t reserved = 0;
};

struct Mono27BWeightEntry {
    char name[64] = {};
    uint32_t ggml_type = 0;
    uint32_t source_model = 0;
    uint32_t role = 0;
    uint32_t row_elems = 0;
    uint32_t row_count = 0;
    uint32_t reserved = 0;
    uint64_t data_offset = 0;
    uint64_t data_bytes = 0;
};

bool mono27b_validate_blob_file(const std::string & path, std::string & error);
bool mono27b_read_blob_header(const std::string & path, Mono27BBlobHeader & header, std::string & error);
bool mono27b_read_blob_sections(const std::string & path,
                                Mono27BBlobSection * sections,
                                size_t section_cap,
                                std::string & error);
bool mono27b_read_tokenizer_section(const std::string & path, Mono27BTokenizerSection & section, std::string & error);
bool mono27b_read_weights_section(const std::string & path,
                                  Mono27BWeightsSection & section,
                                  Mono27BWeightEntry * entries,
                                  size_t entry_cap,
                                  std::string & error);
bool mono27b_scan_weight_entry_by_role(const std::string & path,
                                       uint32_t role,
                                       Mono27BWeightEntry & out,
                                       std::string & error);
bool mono27b_scan_weight_entry_by_source_type(const std::string & path,
                                              uint32_t source,
                                              uint32_t ggml_type,
                                              Mono27BWeightEntry & out,
                                              std::string & error);
bool mono27b_write_stub_blob(const std::string & path,
                             int max_ctx_hint,
                             const Mono27BTokenizerSection & tokenizer,
                             const unsigned char * tokenizer_bytes,
                             uint64_t tokenizer_bytes_size,
                             const Mono27BWeightsSection & weights,
                             const Mono27BWeightEntry * entries,
                             const unsigned char * weight_bytes,
                             uint64_t weight_bytes_size,
                             std::string & error);
