#include "mono27b_packer.h"

#include "mono27b_format.h"
#include "mono27b_gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct PackedTensor {
    std::string name;
    uint32_t ggml_type = 0;
    uint32_t source_model = 0;
    uint32_t role = 0;
    uint32_t row_elems = 0;
    uint32_t row_count = 0;
    std::vector<unsigned char> bytes;
};

struct TensorCandidate {
    int64_t tensor_id = -1;
    std::string name;
    uint32_t row_count = 0;
    uint32_t score = 0;
    uint32_t ggml_type = 0;
};

constexpr uint32_t MONO27B_OUTPUT_HEAD_ROWS = 4096U;

static std::string basename_of(const std::string & path) {
    return std::filesystem::path(path).filename().string();
}

static bool contains_ci(std::string haystack, std::string needle) {
    for (char & ch : haystack) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    for (char & ch : needle) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return haystack.find(needle) != std::string::npos;
}

static std::string lower_copy(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static uint32_t score_tensor_name(const std::string & name) {
    const std::string lower = lower_copy(name);
    uint32_t score = 0;
    if (lower.find("blk.0.") != std::string::npos) {
        score += 100U;
    }
    if (lower.find("attn_q.weight") != std::string::npos) {
        score += 90U;
    } else if (lower.find("attn_k.weight") != std::string::npos ||
               lower.find("attn_v.weight") != std::string::npos ||
               lower.find("attn_o.weight") != std::string::npos) {
        score += 80U;
    } else if (lower.find("ffn_down.weight") != std::string::npos) {
        score += 70U;
    } else if (lower.find("ffn_up.weight") != std::string::npos ||
               lower.find("ffn_gate.weight") != std::string::npos) {
        score += 60U;
    } else if (lower.find("ssm_alpha.weight") != std::string::npos) {
        score += 50U;
    }
    if (lower.find("token_embd") != std::string::npos || lower.find("output.weight") != std::string::npos) {
        score = 0U;
    }
    return score;
}

static bool read_tensor_payload(const std::string & path,
                                const Mono27BGgufFile & gguf,
                                const Mono27BGgufTensorInfo & tensor,
                                std::vector<unsigned char> & out,
                                std::string & error) {
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "failed to open gguf payload: " + path;
        return false;
    }
    out.resize(static_cast<size_t>(tensor.size_bytes));
    const uint64_t off = gguf.data_offset + tensor.offset;
    const bool ok = std::fseek(fp, static_cast<long>(off), SEEK_SET) == 0 &&
                    std::fread(out.data(), 1, out.size(), fp) == out.size();
    std::fclose(fp);
    if (!ok) {
        error = "failed to read matrix tensor bytes: " + tensor.name;
        return false;
    }
    return true;
}

static bool load_matrix_tensor(const std::string & path,
                               const Mono27BGgufFile & gguf,
                               uint32_t wanted_type,
                               uint32_t row_elems,
                               uint32_t source_model,
                               uint32_t role,
                               PackedTensor & out,
                               std::string & error) {
    const size_t row_bytes = mono27b_ggml_row_size(wanted_type, row_elems);
    if (row_elems == 0 || row_bytes == 0) {
        error = "matrix tensor row width is incompatible with quant block size";
        return false;
    }

    TensorCandidate best{};
    for (size_t i = 0; i < gguf.tensors.size(); ++i) {
        const Mono27BGgufTensorInfo & t = gguf.tensors[i];
        if (t.ggml_type != wanted_type || t.size_bytes == 0 || t.size_bytes % row_bytes != 0) {
            continue;
        }
        const uint32_t row_count = static_cast<uint32_t>(t.size_bytes / row_bytes);
        if (row_count == 0 || row_count > 8192U) {
            continue;
        }
        TensorCandidate cand{};
        cand.tensor_id = static_cast<int64_t>(i);
        cand.name = t.name;
        cand.row_count = row_count;
        cand.ggml_type = wanted_type;
        cand.score = score_tensor_name(t.name);
        if (best.tensor_id < 0 || cand.score > best.score ||
            (cand.score == best.score && cand.row_count < best.row_count)) {
            best = cand;
        }
    }
    if (best.tensor_id < 0) {
        error = wanted_type == MONO27B_GGML_TYPE_Q4_K
            ? "failed to find suitable q4_k target tensor"
            : "failed to find suitable q8_0 draft tensor";
        return false;
    }

    const Mono27BGgufTensorInfo & tensor = gguf.tensors[static_cast<size_t>(best.tensor_id)];
    out.name = tensor.name;
    out.ggml_type = tensor.ggml_type;
    out.source_model = source_model;
    out.role = role;
    out.row_elems = row_elems;
    out.row_count = best.row_count;
    return read_tensor_payload(path, gguf, tensor, out.bytes, error);
}

static bool load_named_matrix_tensor(const std::string & path,
                                     const Mono27BGgufFile & gguf,
                                     const char * wanted_name,
                                     uint32_t wanted_type,
                                     uint32_t row_elems,
                                     uint32_t source_model,
                                     uint32_t role,
                                     uint32_t max_rows,
                                     PackedTensor & out,
                                     std::string & error) {
    const size_t row_bytes = mono27b_ggml_row_size(wanted_type, row_elems);
    if (row_elems == 0 || row_bytes == 0) {
        error = "named matrix tensor row width is incompatible with quant block size";
        return false;
    }

    const Mono27BGgufTensorInfo * tensor = nullptr;
    for (const Mono27BGgufTensorInfo & t : gguf.tensors) {
        if (t.name == wanted_name) {
            tensor = &t;
            break;
        }
    }
    if (!tensor) {
        error = std::string("missing named tensor: ") + wanted_name;
        return false;
    }
    if (tensor->ggml_type != wanted_type || tensor->size_bytes == 0 || tensor->size_bytes % row_bytes != 0) {
        error = std::string("named tensor has incompatible quant/layout: ") + wanted_name;
        return false;
    }

    const uint32_t full_row_count = static_cast<uint32_t>(tensor->size_bytes / row_bytes);
    const uint32_t kept_rows = max_rows > 0 ? std::min(full_row_count, max_rows) : full_row_count;
    if (kept_rows == 0) {
        error = std::string("named tensor has no rows: ") + wanted_name;
        return false;
    }

    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "failed to open gguf payload: " + path;
        return false;
    }
    const uint64_t off = gguf.data_offset + tensor->offset;
    const size_t kept_bytes = row_bytes * static_cast<size_t>(kept_rows);
    out.bytes.resize(kept_bytes);
    const bool ok = std::fseek(fp, static_cast<long>(off), SEEK_SET) == 0 &&
                    std::fread(out.bytes.data(), 1, kept_bytes, fp) == kept_bytes;
    std::fclose(fp);
    if (!ok) {
        error = std::string("failed to read named tensor bytes: ") + wanted_name;
        return false;
    }

    out.name = tensor->name;
    out.ggml_type = tensor->ggml_type;
    out.source_model = source_model;
    out.role = role;
    out.row_elems = row_elems;
    out.row_count = kept_rows;
    return true;
}

static bool validate_inputs(const std::string & target_gguf,
                            const std::string & draft_gguf,
                            std::string & error) {
    if (!std::filesystem::exists(target_gguf)) {
        error = "missing target gguf: " + target_gguf;
        return false;
    }
    if (!std::filesystem::exists(draft_gguf)) {
        error = "missing draft gguf: " + draft_gguf;
        return false;
    }
    const std::string target_name = basename_of(target_gguf);
    const std::string draft_name = basename_of(draft_gguf);
    if (!contains_ci(target_name, "q4_k_m")) {
        error = "target gguf must be q4_k_m: " + target_name;
        return false;
    }
    if (!contains_ci(draft_name, "q8_0")) {
        error = "draft gguf must be q8_0: " + draft_name;
        return false;
    }
    return true;
}

} // namespace

bool mono27b_pack_models(const std::string & target_gguf,
                         const std::string & draft_gguf,
                         const std::string & out_blob,
                         std::string & status,
                         std::string & error) {
    if (!validate_inputs(target_gguf, draft_gguf, error)) {
        return false;
    }

    Mono27BGgufFile target{};
    Mono27BGgufFile draft{};
    if (!mono27b_read_gguf(target_gguf, target, error) ||
        !mono27b_read_gguf(draft_gguf, draft, error)) {
        return false;
    }

    const Mono27BGgufMetadata & target_meta = target.metadata;
    const Mono27BGgufMetadata & draft_meta = draft.metadata;
    if (target_meta.architecture != MONO27B_TARGET_ARCH) {
        error = "target gguf architecture mismatch: " + target_meta.architecture;
        return false;
    }
    if (draft_meta.architecture != MONO27B_DRAFT_ARCH && draft_meta.architecture != "qwen3") {
        error = "draft gguf architecture mismatch: " + draft_meta.architecture;
        return false;
    }
    if (target_meta.embedding_length != MONO27B_TARGET_HIDDEN ||
        target_meta.block_count != MONO27B_TARGET_LAYERS ||
        target_meta.feed_forward_length != MONO27B_TARGET_FFN ||
        target_meta.head_count != MONO27B_TARGET_HEADS ||
        target_meta.head_count_kv != MONO27B_TARGET_KV_HEADS) {
        char buf[256] = {};
        std::snprintf(buf,
                      sizeof(buf),
                      "target gguf hparams mismatch: emb=%u layers=%u ff=%u heads=%u kv_heads=%u",
                      target_meta.embedding_length,
                      target_meta.block_count,
                      target_meta.feed_forward_length,
                      target_meta.head_count,
                      target_meta.head_count_kv);
        error = buf;
        return false;
    }
    if (draft_meta.embedding_length == 0U) {
        error = "draft gguf missing embedding_length metadata";
        return false;
    }

    Mono27BTokenizerSection tokenizer{};
    tokenizer.magic = MONO27B_TOKENIZER_MAGIC;
    tokenizer.version = 1U;
    tokenizer.vocab_size = target_meta.vocab_size;
    tokenizer.merges_count = target_meta.merges_count;
    tokenizer.bos_id = target_meta.bos_id;
    tokenizer.eos_id = target_meta.eos_id;
    tokenizer.im_start_id = target_meta.im_start_id;
    tokenizer.im_end_id = target_meta.im_end_id;

    std::vector<unsigned char> tokenizer_bytes;
    tokenizer_bytes.reserve(target_meta.vocab_size * 8U);
    for (const std::string & tok : target_meta.tokens) {
        const uint32_t len = static_cast<uint32_t>(tok.size());
        const size_t old = tokenizer_bytes.size();
        tokenizer_bytes.resize(old + sizeof(uint32_t) + len);
        std::memcpy(tokenizer_bytes.data() + old, &len, sizeof(uint32_t));
        if (len > 0) {
            std::memcpy(tokenizer_bytes.data() + old + sizeof(uint32_t), tok.data(), len);
        }
    }
    for (const std::string & merge : target_meta.merges) {
        const uint32_t len = static_cast<uint32_t>(merge.size());
        const size_t old = tokenizer_bytes.size();
        tokenizer_bytes.resize(old + sizeof(uint32_t) + len);
        std::memcpy(tokenizer_bytes.data() + old, &len, sizeof(uint32_t));
        if (len > 0) {
            std::memcpy(tokenizer_bytes.data() + old + sizeof(uint32_t), merge.data(), len);
        }
    }

    PackedTensor target_tensor;
    PackedTensor draft_tensor;
    PackedTensor output_head_tensor;
    if (!load_named_matrix_tensor(target_gguf, target, "token_embd.weight", MONO27B_GGML_TYPE_Q4_K, target_meta.embedding_length,
                                  MONO27B_SOURCE_TARGET, MONO27B_ROLE_TARGET_MATVEC, 0U,
                                  target_tensor, error) ||
        !load_matrix_tensor(draft_gguf, draft, MONO27B_GGML_TYPE_Q8_0, draft_meta.embedding_length,
                            MONO27B_SOURCE_DRAFT, MONO27B_ROLE_DRAFT_MATVEC, draft_tensor, error) ||
        !load_named_matrix_tensor(target_gguf, target, "output.weight", MONO27B_GGML_TYPE_Q6_K, target_meta.embedding_length,
                                  MONO27B_SOURCE_TARGET, MONO27B_ROLE_TARGET_OUTPUT_HEAD, MONO27B_OUTPUT_HEAD_ROWS,
                                  output_head_tensor, error)) {
        return false;
    }

    Mono27BWeightsSection weights{};
    weights.magic = MONO27B_WEIGHTS_MAGIC;
    weights.version = 1U;
    weights.entry_count = 3U;

    Mono27BWeightEntry entries[3] = {};
    const uint64_t payload_base = sizeof(Mono27BWeightsSection) + 3U * sizeof(Mono27BWeightEntry);

    std::snprintf(entries[0].name, sizeof(entries[0].name), "%s", target_tensor.name.c_str());
    entries[0].ggml_type = target_tensor.ggml_type;
    entries[0].source_model = target_tensor.source_model;
    entries[0].role = target_tensor.role;
    entries[0].row_elems = target_tensor.row_elems;
    entries[0].row_count = target_tensor.row_count;
    entries[0].data_offset = payload_base;
    entries[0].data_bytes = static_cast<uint64_t>(target_tensor.bytes.size());

    std::snprintf(entries[1].name, sizeof(entries[1].name), "%s", draft_tensor.name.c_str());
    entries[1].ggml_type = draft_tensor.ggml_type;
    entries[1].source_model = draft_tensor.source_model;
    entries[1].role = draft_tensor.role;
    entries[1].row_elems = draft_tensor.row_elems;
    entries[1].row_count = draft_tensor.row_count;
    entries[1].data_offset = payload_base + entries[0].data_bytes;
    entries[1].data_bytes = static_cast<uint64_t>(draft_tensor.bytes.size());

    std::snprintf(entries[2].name, sizeof(entries[2].name), "%s", output_head_tensor.name.c_str());
    entries[2].ggml_type = output_head_tensor.ggml_type;
    entries[2].source_model = output_head_tensor.source_model;
    entries[2].role = output_head_tensor.role;
    entries[2].row_elems = output_head_tensor.row_elems;
    entries[2].row_count = output_head_tensor.row_count;
    entries[2].data_offset = entries[1].data_offset + entries[1].data_bytes;
    entries[2].data_bytes = static_cast<uint64_t>(output_head_tensor.bytes.size());

    std::vector<unsigned char> weight_bytes(target_tensor.bytes.size() + draft_tensor.bytes.size() + output_head_tensor.bytes.size());
    std::memcpy(weight_bytes.data(), target_tensor.bytes.data(), target_tensor.bytes.size());
    std::memcpy(weight_bytes.data() + target_tensor.bytes.size(), draft_tensor.bytes.data(), draft_tensor.bytes.size());
    std::memcpy(weight_bytes.data() + target_tensor.bytes.size() + draft_tensor.bytes.size(),
                output_head_tensor.bytes.data(),
                output_head_tensor.bytes.size());

    if (!mono27b_write_stub_blob(out_blob,
                                 131072,
                                 tokenizer,
                                 tokenizer_bytes.data(),
                                 static_cast<uint64_t>(tokenizer_bytes.size()),
                                 weights,
                                 entries,
                                 weight_bytes.data(),
                                 static_cast<uint64_t>(weight_bytes.size()),
                                 error)) {
        return false;
    }

    char buf[512] = {};
    std::snprintf(buf,
                  sizeof(buf),
                  "wrote blob: %s (target tensor=%s rows=%u, draft tensor=%s rows=%u, output tensor=%s rows=%u)",
                  out_blob.c_str(),
                  target_tensor.name.c_str(),
                  target_tensor.row_count,
                  draft_tensor.name.c_str(),
                  draft_tensor.row_count,
                  output_head_tensor.name.c_str(),
                  output_head_tensor.row_count);
    status = buf;
    return true;
}
