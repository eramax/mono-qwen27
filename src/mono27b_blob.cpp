#include "mono27b_format.h"

#include <cstring>
#include <cstdio>
#include <filesystem>

namespace {

static bool header_strings_match(const Mono27BBlobHeader & header,
                                 const char * expected,
                                 const char * actual) {
    return std::strncmp(expected, actual, 15) == 0;
}

static bool read_at(std::FILE * fp, void * out, size_t bytes, uint64_t off) {
    if (std::fseek(fp, static_cast<long>(off), SEEK_SET) != 0) {
        return false;
    }
    return std::fread(out, bytes, 1, fp) == 1;
}

} // namespace

bool mono27b_read_blob_header(const std::string & path, Mono27BBlobHeader & header, std::string & error) {
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }

    if (std::fread(&header, sizeof(header), 1, fp) != 1) {
        std::fclose(fp);
        error = "invalid blob header: " + path;
        return false;
    }

    std::fclose(fp);
    return true;
}

bool mono27b_validate_blob_file(const std::string & path, std::string & error) {
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(path, header, error)) {
        return false;
    }

    std::error_code ec;
    const uint64_t actual_bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        error = "invalid blob file size: " + path;
        return false;
    }

    if (header.magic != MONO27B_BLOB_MAGIC) {
        error = "invalid blob magic: " + path;
        return false;
    }

    if (header.schema_version != MONO27B_SCHEMA_VERSION) {
        error = "schema mismatch: " + path;
        return false;
    }

    if (header.file_bytes != actual_bytes) {
        error = "invalid blob size metadata: " + path;
        return false;
    }
    if (header.section_count < 2) {
        error = "invalid blob section count: " + path;
        return false;
    }

    if (header.target_layers != MONO27B_TARGET_LAYERS || header.draft_layers != MONO27B_DRAFT_LAYERS) {
        error = "invalid blob layer counts: " + path;
        return false;
    }

    if (!header_strings_match(header, MONO27B_TARGET_ARCH, header.target_arch) ||
        !header_strings_match(header, MONO27B_DRAFT_ARCH, header.draft_arch)) {
        error = "invalid blob architecture tags: " + path;
        return false;
    }

    if (!header_strings_match(header, MONO27B_TARGET_QUANT, header.target_quant) ||
        !header_strings_match(header, MONO27B_DRAFT_QUANT, header.draft_quant)) {
        error = "invalid blob quant tags: " + path;
        return false;
    }

    if (header.tokenizer_offset == 0 || header.tokenizer_bytes == 0) {
        error = "missing tokenizer section in blob";
        return false;
    }

    if (header.tokenizer_offset + header.tokenizer_bytes > header.file_bytes) {
        error = "invalid tokenizer section bounds: " + path;
        return false;
    }

    Mono27BBlobSection sections[4] = {};
    if (!mono27b_read_blob_sections(path, sections, 4, error)) {
        return false;
    }
    bool found_tokenizer = false;
    bool found_weights = false;
    Mono27BBlobSection weights_section{};
    for (uint32_t i = 0; i < header.section_count && i < 4; ++i) {
        if (sections[i].kind == MONO27B_SECTION_TOKENIZER) {
            found_tokenizer = true;
            if (sections[i].offset != header.tokenizer_offset || sections[i].bytes != header.tokenizer_bytes) {
                error = "tokenizer section/header mismatch: " + path;
                return false;
            }
        }
        if (sections[i].kind == MONO27B_SECTION_WEIGHTS) {
            found_weights = true;
            weights_section = sections[i];
            if (sections[i].offset + sections[i].bytes > header.file_bytes) {
                error = "invalid weights section bounds: " + path;
                return false;
            }
        }
    }
    if (!found_tokenizer || !found_weights) {
        error = "missing required blob sections: " + path;
        return false;
    }

    Mono27BTokenizerSection tokenizer{};
    if (!mono27b_read_tokenizer_section(path, tokenizer, error)) {
        return false;
    }

    Mono27BWeightEntry entry_probe{};
    if (!mono27b_scan_weight_entry_by_role(path, MONO27B_ROLE_TARGET_MATVEC, entry_probe, error)) {
        error = "missing target matvec entry: " + path;
        return false;
    }
    if (!mono27b_scan_weight_entry_by_role(path, MONO27B_ROLE_TARGET_OUTPUT_HEAD, entry_probe, error)) {
        error = "missing output head entry: " + path;
        return false;
    }

    return true;
}

bool mono27b_read_blob_sections(const std::string & path,
                                Mono27BBlobSection * sections,
                                size_t section_cap,
                                std::string & error) {
    if (!sections || section_cap == 0) {
        error = "invalid blob section buffer";
        return false;
    }
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(path, header, error)) {
        return false;
    }
    if (header.section_count > section_cap) {
        error = "insufficient blob section buffer: " + path;
        return false;
    }
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    const size_t bytes = static_cast<size_t>(header.section_count) * sizeof(Mono27BBlobSection);
    const bool ok = read_at(fp, sections, bytes, header.section_table_offset);
    std::fclose(fp);
    if (!ok) {
        error = "invalid blob section table read: " + path;
        return false;
    }
    return true;
}

bool mono27b_read_tokenizer_section(const std::string & path, Mono27BTokenizerSection & section, std::string & error) {
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(path, header, error)) {
        return false;
    }
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    const bool ok = read_at(fp, &section, sizeof(section), header.tokenizer_offset);
    std::fclose(fp);
    if (!ok) {
        error = "invalid tokenizer section read: " + path;
        return false;
    }
    if (section.magic != MONO27B_TOKENIZER_MAGIC) {
        error = "invalid tokenizer section magic: " + path;
        return false;
    }
    if (section.version != 1U) {
        error = "invalid tokenizer section version: " + path;
        return false;
    }
    if (section.vocab_size == 0) {
        error = "invalid tokenizer vocab size: " + path;
        return false;
    }
    return true;
}

bool mono27b_read_weights_section(const std::string & path,
                                  Mono27BWeightsSection & section,
                                  Mono27BWeightEntry * entries,
                                  size_t entry_cap,
                                  std::string & error) {
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(path, header, error)) {
        return false;
    }
    Mono27BBlobSection sections[4] = {};
    if (!mono27b_read_blob_sections(path, sections, 4, error)) {
        return false;
    }
    Mono27BBlobSection weights_blob{};
    bool found = false;
    for (uint32_t i = 0; i < header.section_count && i < 4; ++i) {
        if (sections[i].kind == MONO27B_SECTION_WEIGHTS) {
            weights_blob = sections[i];
            found = true;
            break;
        }
    }
    if (!found) {
        error = "missing weights section: " + path;
        return false;
    }
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    if (!read_at(fp, &section, sizeof(section), weights_blob.offset)) {
        std::fclose(fp);
        error = "invalid weights section read: " + path;
        return false;
    }
    if (section.magic != MONO27B_WEIGHTS_MAGIC || section.version != 1U) {
        std::fclose(fp);
        error = "invalid weights section header: " + path;
        return false;
    }
    if (section.entry_count == 0) {
        std::fclose(fp);
        error = "invalid weights entry count: " + path;
        return false;
    }
    const uint32_t to_read = entry_cap > 0
        ? static_cast<uint32_t>(std::min(static_cast<size_t>(section.entry_count), entry_cap))
        : 0U;
    const uint64_t entries_off = weights_blob.offset + sizeof(Mono27BWeightsSection);
    if (to_read > 0) {
        const size_t entries_bytes = static_cast<size_t>(to_read) * sizeof(Mono27BWeightEntry);
        if (!read_at(fp, entries, entries_bytes, entries_off)) {
            std::fclose(fp);
            error = "invalid weights entries read: " + path;
            return false;
        }
    }
    for (uint32_t i = 0; i < to_read; ++i) {
        const uint64_t abs_off = weights_blob.offset + entries[i].data_offset;
        if (entries[i].data_bytes == 0 || abs_off + entries[i].data_bytes > weights_blob.offset + weights_blob.bytes) {
            std::fclose(fp);
            error = "invalid weights entry bounds: " + path;
            return false;
        }
        if ((entries[i].role == MONO27B_ROLE_TARGET_MATVEC ||
             entries[i].role == MONO27B_ROLE_DRAFT_MATVEC ||
             entries[i].role == MONO27B_ROLE_TARGET_OUTPUT_HEAD) &&
            (entries[i].row_elems == 0 || entries[i].row_count == 0)) {
            std::fclose(fp);
            error = "invalid matrix entry metadata: " + path;
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

static bool find_weights_section_offset(const std::string & path, uint64_t & weights_off, std::string & error) {
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(path, header, error)) {
        return false;
    }
    Mono27BBlobSection sections[8] = {};
    const uint32_t n = std::min(header.section_count, 8U);
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(Mono27BBlobSection);
    const bool ok = read_at(fp, sections, bytes, header.section_table_offset);
    std::fclose(fp);
    if (!ok) {
        error = "invalid blob section table: " + path;
        return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (sections[i].kind == MONO27B_SECTION_WEIGHTS) {
            weights_off = sections[i].offset;
            return true;
        }
    }
    error = "missing weights section: " + path;
    return false;
}

bool mono27b_scan_weight_entry_by_role(const std::string & path,
                                       uint32_t role,
                                       Mono27BWeightEntry & out,
                                       std::string & error) {
    uint64_t weights_off = 0;
    if (!find_weights_section_offset(path, weights_off, error)) {
        return false;
    }
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    Mono27BWeightsSection section{};
    if (!read_at(fp, &section, sizeof(section), weights_off) ||
        section.magic != MONO27B_WEIGHTS_MAGIC || section.entry_count == 0) {
        std::fclose(fp);
        error = "invalid weights section: " + path;
        return false;
    }
    const uint64_t entries_off = weights_off + sizeof(Mono27BWeightsSection);
    if (std::fseek(fp, static_cast<long>(entries_off), SEEK_SET) != 0) {
        std::fclose(fp);
        error = "failed to seek weight entries: " + path;
        return false;
    }
    for (uint32_t i = 0; i < section.entry_count; ++i) {
        Mono27BWeightEntry entry{};
        if (std::fread(&entry, sizeof(entry), 1, fp) != 1) {
            std::fclose(fp);
            error = "failed to read weight entry: " + path;
            return false;
        }
        if (entry.role == role) {
            out = entry;
            std::fclose(fp);
            return true;
        }
    }
    std::fclose(fp);
    error = "no entry with role " + std::to_string(role) + ": " + path;
    return false;
}

bool mono27b_scan_weight_entry_by_source_type(const std::string & path,
                                              uint32_t source,
                                              uint32_t ggml_type,
                                              Mono27BWeightEntry & out,
                                              std::string & error) {
    uint64_t weights_off = 0;
    if (!find_weights_section_offset(path, weights_off, error)) {
        return false;
    }
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "missing blob: " + path;
        return false;
    }
    Mono27BWeightsSection section{};
    if (!read_at(fp, &section, sizeof(section), weights_off) ||
        section.magic != MONO27B_WEIGHTS_MAGIC || section.entry_count == 0) {
        std::fclose(fp);
        error = "invalid weights section: " + path;
        return false;
    }
    const uint64_t entries_off = weights_off + sizeof(Mono27BWeightsSection);
    if (std::fseek(fp, static_cast<long>(entries_off), SEEK_SET) != 0) {
        std::fclose(fp);
        error = "failed to seek weight entries: " + path;
        return false;
    }
    for (uint32_t i = 0; i < section.entry_count; ++i) {
        Mono27BWeightEntry entry{};
        if (std::fread(&entry, sizeof(entry), 1, fp) != 1) {
            std::fclose(fp);
            error = "failed to read weight entry: " + path;
            return false;
        }
        if (entry.source_model == source && entry.ggml_type == ggml_type && entry.row_elems > 0 && entry.row_count > 0) {
            out = entry;
            std::fclose(fp);
            return true;
        }
    }
    std::fclose(fp);
    error = "no entry with source=" + std::to_string(source) + " type=" + std::to_string(ggml_type) + ": " + path;
    return false;
}

bool mono27b_write_stub_blob(const std::string & path,
                             int max_ctx_hint,
                             const Mono27BTokenizerSection & tokenizer,
                             const unsigned char * tokenizer_bytes,
                             uint64_t tokenizer_bytes_size,
                             const Mono27BWeightsSection & weights,
                             const Mono27BWeightEntry * entries,
                             const unsigned char * weight_bytes,
                             uint64_t weight_bytes_size,
                             std::string & error) {
    Mono27BBlobHeader header{};
    header.section_count = 2;
    const uint64_t section_table_offset = sizeof(Mono27BBlobHeader);
    const uint64_t tokenizer_offset = section_table_offset + 2 * sizeof(Mono27BBlobSection);
    const uint64_t tokenizer_total_bytes = sizeof(Mono27BTokenizerSection) + tokenizer_bytes_size;
    const uint64_t weights_offset = tokenizer_offset + tokenizer_total_bytes;
    const uint64_t weights_payload_offset = weights_offset + sizeof(Mono27BWeightsSection) +
        static_cast<uint64_t>(weights.entry_count) * sizeof(Mono27BWeightEntry);
    const Mono27BBlobSection tokenizer_descriptor{
        MONO27B_SECTION_TOKENIZER,
        0,
        tokenizer_offset,
        tokenizer_total_bytes,
    };
    const Mono27BBlobSection weights_descriptor{
        MONO27B_SECTION_WEIGHTS,
        0,
        weights_offset,
        (weights_payload_offset - weights_offset) + weight_bytes_size,
    };
    header.section_table_offset = sizeof(Mono27BBlobHeader);
    header.tokenizer_offset = tokenizer_descriptor.offset;
    header.tokenizer_bytes = tokenizer_descriptor.bytes;
    header.max_ctx_hint = static_cast<uint32_t>(max_ctx_hint);
    header.file_bytes = weights_descriptor.offset + weights_descriptor.bytes;
    std::snprintf(header.target_arch, sizeof(header.target_arch), "%s", MONO27B_TARGET_ARCH);
    std::snprintf(header.draft_arch, sizeof(header.draft_arch), "%s", MONO27B_DRAFT_ARCH);
    std::snprintf(header.target_quant, sizeof(header.target_quant), "%s", MONO27B_TARGET_QUANT);
    std::snprintf(header.draft_quant, sizeof(header.draft_quant), "%s", MONO27B_DRAFT_QUANT);

    std::FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        error = "failed to create blob: " + path;
        return false;
    }

    const bool ok = std::fwrite(&header, sizeof(header), 1, fp) == 1 &&
                    std::fwrite(&tokenizer_descriptor, sizeof(tokenizer_descriptor), 1, fp) == 1 &&
                    std::fwrite(&weights_descriptor, sizeof(weights_descriptor), 1, fp) == 1 &&
                    std::fwrite(&tokenizer, sizeof(tokenizer), 1, fp) == 1 &&
                    std::fwrite(tokenizer_bytes, 1, static_cast<size_t>(tokenizer_bytes_size), fp) == tokenizer_bytes_size &&
                    std::fwrite(&weights, sizeof(weights), 1, fp) == 1 &&
                    std::fwrite(entries, sizeof(Mono27BWeightEntry), weights.entry_count, fp) == weights.entry_count &&
                    std::fwrite(weight_bytes, 1, static_cast<size_t>(weight_bytes_size), fp) == weight_bytes_size;
    std::fclose(fp);
    if (!ok) {
        error = "failed to write blob: " + path;
        return false;
    }
    return true;
}
