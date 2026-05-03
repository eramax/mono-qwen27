#include "mono27b_gguf.h"
#include "mono27b_format.h"

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct KvEntry {
    uint32_t type = 0;
    uint32_t arr_type = 0;
    uint64_t arr_n = 0;
    uint32_t u32 = 0;
    std::string str;
    std::vector<std::string> arr_str;
};

static bool read_exact(std::FILE * fp, void * dst, size_t n, std::string & error, const char * what) {
    if (std::fread(dst, 1, n, fp) != n) {
        error = std::string("gguf read failed: ") + what;
        return false;
    }
    return true;
}

template <typename T>
static bool read_scalar(std::FILE * fp, T & dst, std::string & error, const char * what) {
    return read_exact(fp, &dst, sizeof(T), error, what);
}

static bool read_string(std::FILE * fp, std::string & out, std::string & error, const char * what) {
    uint64_t n = 0;
    if (!read_scalar(fp, n, error, what)) {
        return false;
    }
    out.resize(static_cast<size_t>(n));
    if (n == 0) {
        return true;
    }
    return read_exact(fp, out.data(), static_cast<size_t>(n), error, what);
}

static bool skip_bytes(std::FILE * fp, uint64_t n, std::string & error, const char * what) {
    if (std::fseek(fp, static_cast<long>(n), SEEK_CUR) != 0) {
        error = std::string("gguf seek failed: ") + what;
        return false;
    }
    return true;
}

static bool skip_value(std::FILE * fp, uint32_t type, std::string & error);

static bool skip_array(std::FILE * fp, std::string & error) {
    uint32_t arr_type = 0;
    uint64_t n = 0;
    if (!read_scalar(fp, arr_type, error, "gguf array type") ||
        !read_scalar(fp, n, error, "gguf array size")) {
        return false;
    }
    if (arr_type == MONO27B_GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < n; ++i) {
            std::string s;
            if (!read_string(fp, s, error, "gguf array string")) {
                return false;
            }
        }
        return true;
    }
    const size_t elem_size =
        arr_type == MONO27B_GGUF_TYPE_UINT8 || arr_type == MONO27B_GGUF_TYPE_INT8 || arr_type == MONO27B_GGUF_TYPE_BOOL ? 1 :
        arr_type == MONO27B_GGUF_TYPE_UINT16 || arr_type == MONO27B_GGUF_TYPE_INT16 ? 2 :
        arr_type == MONO27B_GGUF_TYPE_UINT32 || arr_type == MONO27B_GGUF_TYPE_INT32 || arr_type == MONO27B_GGUF_TYPE_FLOAT32 ? 4 :
        arr_type == MONO27B_GGUF_TYPE_UINT64 || arr_type == MONO27B_GGUF_TYPE_INT64 || arr_type == MONO27B_GGUF_TYPE_FLOAT64 ? 8 : 0;
    if (elem_size == 0) {
        error = "unsupported gguf array element type";
        return false;
    }
    return skip_bytes(fp, static_cast<uint64_t>(elem_size) * n, error, "gguf array payload");
}

static bool skip_value(std::FILE * fp, uint32_t type, std::string & error) {
    switch (type) {
        case MONO27B_GGUF_TYPE_UINT8:
        case MONO27B_GGUF_TYPE_INT8:
        case MONO27B_GGUF_TYPE_BOOL:
            return skip_bytes(fp, 1, error, "gguf scalar 1");
        case MONO27B_GGUF_TYPE_UINT16:
        case MONO27B_GGUF_TYPE_INT16:
            return skip_bytes(fp, 2, error, "gguf scalar 2");
        case MONO27B_GGUF_TYPE_UINT32:
        case MONO27B_GGUF_TYPE_INT32:
        case MONO27B_GGUF_TYPE_FLOAT32:
            return skip_bytes(fp, 4, error, "gguf scalar 4");
        case MONO27B_GGUF_TYPE_UINT64:
        case MONO27B_GGUF_TYPE_INT64:
        case MONO27B_GGUF_TYPE_FLOAT64:
            return skip_bytes(fp, 8, error, "gguf scalar 8");
        case MONO27B_GGUF_TYPE_STRING: {
            std::string s;
            return read_string(fp, s, error, "gguf scalar string");
        }
        case MONO27B_GGUF_TYPE_ARRAY:
            return skip_array(fp, error);
        default:
            error = "unsupported gguf value type";
            return false;
    }
}

static bool read_kv_value(std::FILE * fp, KvEntry & kv, std::string & error) {
    if (kv.type == MONO27B_GGUF_TYPE_UINT32) {
        return read_scalar(fp, kv.u32, error, "gguf u32");
    }
    if (kv.type == MONO27B_GGUF_TYPE_STRING) {
        return read_string(fp, kv.str, error, "gguf string");
    }
    if (kv.type == MONO27B_GGUF_TYPE_ARRAY) {
        if (!read_scalar(fp, kv.arr_type, error, "gguf array type") ||
            !read_scalar(fp, kv.arr_n, error, "gguf array n")) {
            return false;
        }
        if (kv.arr_type == MONO27B_GGUF_TYPE_STRING) {
            kv.arr_str.reserve(static_cast<size_t>(kv.arr_n));
            for (uint64_t i = 0; i < kv.arr_n; ++i) {
                std::string s;
                if (!read_string(fp, s, error, "gguf array string")) {
                    return false;
                }
                kv.arr_str.push_back(std::move(s));
            }
            return true;
        }
        const size_t elem_size =
            kv.arr_type == MONO27B_GGUF_TYPE_UINT8 || kv.arr_type == MONO27B_GGUF_TYPE_INT8 || kv.arr_type == MONO27B_GGUF_TYPE_BOOL ? 1 :
            kv.arr_type == MONO27B_GGUF_TYPE_UINT16 || kv.arr_type == MONO27B_GGUF_TYPE_INT16 ? 2 :
            kv.arr_type == MONO27B_GGUF_TYPE_UINT32 || kv.arr_type == MONO27B_GGUF_TYPE_INT32 || kv.arr_type == MONO27B_GGUF_TYPE_FLOAT32 ? 4 :
            kv.arr_type == MONO27B_GGUF_TYPE_UINT64 || kv.arr_type == MONO27B_GGUF_TYPE_INT64 || kv.arr_type == MONO27B_GGUF_TYPE_FLOAT64 ? 8 : 0;
        if (elem_size == 0) {
            error = "unsupported gguf array element type";
            return false;
        }
        return skip_bytes(fp, static_cast<uint64_t>(elem_size) * kv.arr_n, error, "gguf array payload");
    }
    return skip_value(fp, kv.type, error);
}

static uint64_t align_u64(uint64_t x, uint64_t align) {
    const uint64_t mask = align - 1U;
    return (x + mask) & ~mask;
}

static size_t type_size(uint32_t ggml_type) {
    switch (ggml_type) {
        case MONO27B_GGML_TYPE_Q8_0: return 34;
        case MONO27B_GGML_TYPE_Q4_K: return 144;
        case MONO27B_GGML_TYPE_Q6_K: return 210;
        default: return 0;
    }
}

static size_t block_size(uint32_t ggml_type) {
    switch (ggml_type) {
        case MONO27B_GGML_TYPE_Q8_0: return 32;
        case MONO27B_GGML_TYPE_Q4_K: return 256;
        case MONO27B_GGML_TYPE_Q6_K: return 256;
        default: return 0;
    }
}

static void fill_metadata(const std::unordered_map<std::string, KvEntry> & kvs,
                          Mono27BGgufMetadata & out) {
    auto get_u32 = [&](const std::string & key, uint32_t def = 0) -> uint32_t {
        const auto it = kvs.find(key);
        return it != kvs.end() && it->second.type == MONO27B_GGUF_TYPE_UINT32 ? it->second.u32 : def;
    };
    auto get_str = [&](const std::string & key) -> std::string {
        const auto it = kvs.find(key);
        return it != kvs.end() && it->second.type == MONO27B_GGUF_TYPE_STRING ? it->second.str : std::string();
    };
    const auto it_tokens = kvs.find("tokenizer.ggml.tokens");
    if (it_tokens != kvs.end()) {
        out.tokens = it_tokens->second.arr_str;
        out.vocab_size = static_cast<uint32_t>(it_tokens->second.arr_str.size());
        for (size_t i = 0; i < it_tokens->second.arr_str.size(); ++i) {
            if (it_tokens->second.arr_str[i] == "<|im_start|>") {
                out.im_start_id = static_cast<uint32_t>(i);
            }
            if (it_tokens->second.arr_str[i] == "<|im_end|>") {
                out.im_end_id = static_cast<uint32_t>(i);
            }
        }
    }
    const auto it_merges = kvs.find("tokenizer.ggml.merges");
    if (it_merges != kvs.end()) {
        out.merges = it_merges->second.arr_str;
        out.merges_count = static_cast<uint32_t>(it_merges->second.arr_str.size());
    }

    out.architecture = get_str("general.architecture");
    out.bos_id = get_u32("tokenizer.ggml.bos_token_id");
    out.eos_id = get_u32("tokenizer.ggml.eos_token_id");
    const std::string prefix = out.architecture == MONO27B_DRAFT_ARCH ? MONO27B_DRAFT_ARCH : MONO27B_TARGET_ARCH;
    out.embedding_length = get_u32(prefix + ".embedding_length");
    out.block_count = get_u32(prefix + ".block_count");
    out.feed_forward_length = get_u32(prefix + ".feed_forward_length");
    out.head_count = get_u32(prefix + ".attention.head_count");
    out.head_count_kv = get_u32(prefix + ".attention.head_count_kv");
}

} // namespace

bool mono27b_read_gguf(const std::string & path, Mono27BGgufFile & out, std::string & error) {
    std::FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "failed to open gguf: " + path;
        return false;
    }

    char magic[4] = {};
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    if (!read_exact(fp, magic, sizeof(magic), error, "gguf magic") ||
        !read_scalar(fp, version, error, "gguf version") ||
        !read_scalar(fp, n_tensors, error, "gguf n_tensors") ||
        !read_scalar(fp, n_kv, error, "gguf n_kv")) {
        std::fclose(fp);
        return false;
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        std::fclose(fp);
        error = "invalid gguf magic: " + path;
        return false;
    }
    if (version != MONO27B_GGUF_VERSION) {
        std::fclose(fp);
        error = "unsupported gguf version: " + path;
        return false;
    }

    std::unordered_map<std::string, KvEntry> kvs;
    kvs.reserve(static_cast<size_t>(n_kv));
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        KvEntry kv{};
        if (!read_string(fp, key, error, "gguf key") ||
            !read_scalar(fp, kv.type, error, "gguf value type") ||
            !read_kv_value(fp, kv, error)) {
            std::fclose(fp);
            return false;
        }
        kvs.emplace(std::move(key), std::move(kv));
    }

    uint32_t alignment = MONO27B_GGUF_DEFAULT_ALIGNMENT;
    const auto it_align = kvs.find("general.alignment");
    if (it_align != kvs.end() && it_align->second.type == MONO27B_GGUF_TYPE_UINT32 && it_align->second.u32 != 0U) {
        alignment = it_align->second.u32;
    }

    std::vector<Mono27BGgufTensorInfo> tensors;
    tensors.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Mono27BGgufTensorInfo tensor{};
        uint32_t n_dims = 0;
        if (!read_string(fp, tensor.name, error, "gguf tensor name") ||
            !read_scalar(fp, n_dims, error, "gguf tensor dims")) {
            std::fclose(fp);
            return false;
        }
        uint64_t nelements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            if (!read_scalar(fp, dim, error, "gguf tensor dim")) {
                std::fclose(fp);
                return false;
            }
            nelements *= dim;
        }
        if (!read_scalar(fp, tensor.ggml_type, error, "gguf tensor type") ||
            !read_scalar(fp, tensor.offset, error, "gguf tensor offset")) {
            std::fclose(fp);
            return false;
        }
        const size_t tsize = type_size(tensor.ggml_type);
        const size_t bsize = block_size(tensor.ggml_type);
        if (tsize == 0 || bsize == 0 || nelements % bsize != 0) {
            tensor.size_bytes = 0;
            tensors.push_back(std::move(tensor));
            continue;
        }
        tensor.size_bytes = static_cast<uint64_t>(nelements / bsize) * static_cast<uint64_t>(tsize);
        tensors.push_back(std::move(tensor));
    }

    const long meta_end = std::ftell(fp);
    if (meta_end < 0) {
        std::fclose(fp);
        error = "failed to determine gguf data offset";
        return false;
    }
    out.version = version;
    out.alignment = alignment;
    out.data_offset = align_u64(static_cast<uint64_t>(meta_end), alignment);
    out.tensors = std::move(tensors);
    fill_metadata(kvs, out.metadata);
    std::fclose(fp);
    return true;
}

size_t mono27b_ggml_row_size(uint32_t ggml_type, uint32_t ne) {
    const size_t tsize = type_size(ggml_type);
    const size_t bsize = block_size(ggml_type);
    if (tsize == 0 || bsize == 0 || ne == 0 || (ne % bsize) != 0U) {
        return 0;
    }
    return (static_cast<size_t>(ne) / bsize) * tsize;
}
