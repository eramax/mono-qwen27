#include "mono27b_cli.h"
#include "mono27b_config.h"
#include "mono27b_format.h"
#include "mono27b_gguf.h"
#include "mono27b_packer.h"
#include "mono27b_tokenizer.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {

static std::string sanitize_text(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (unsigned char ch : in) {
        if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (ch == '\t') {
            out += "\\t";
        } else if (std::isprint(ch)) {
            out += static_cast<char>(ch);
        } else {
            char buf[5] = {};
            std::snprintf(buf, sizeof(buf), "\\x%02x", static_cast<unsigned>(ch));
            out += buf;
        }
    }
    return out;
}

static bool looks_readable(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    size_t printable = 0;
    size_t alnum_or_space = 0;
    size_t alpha = 0;
    size_t spaces = 0;
    for (unsigned char ch : text) {
        if (ch >= 0x80) {
            return false;
        }
        if (std::isprint(ch) || ch == '\n' || ch == '\r' || ch == '\t') {
            ++printable;
        }
        if (std::isalnum(ch) || std::isspace(ch)) {
            ++alnum_or_space;
        }
        if (std::isalpha(ch)) {
            ++alpha;
        }
        if (std::isspace(ch)) {
            ++spaces;
        }
    }
    return printable * 10 >= text.size() * 9 &&
           alnum_or_space * 10 >= text.size() * 7 &&
           alpha * 10 >= text.size() * 5 &&
           (text.size() < 12 || spaces > 0);
}

}

int main(int argc, char ** argv) {
    Mono27BChatArgs args;
    if (!mono27b_parse_chat_args(argc, argv, args) || args.show_help) {
        mono27b_print_chat_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }
    std::string error;
    std::string blob_path = args.blob_path;
    std::filesystem::path temp_blob_path;
    if (blob_path.empty()) {
        char tmp_pattern[] = "/tmp/mono27b-XXXXXX.m27b";
        const int fd = mkstemps(tmp_pattern, 5);
        if (fd < 0) {
            std::fprintf(stderr, "failed to allocate temp blob path\n");
            return 1;
        }
        std::fclose(fdopen(fd, "wb"));
        temp_blob_path = tmp_pattern;
        std::string status;
        if (!mono27b_pack_models(args.target_gguf, args.draft_gguf, temp_blob_path.string(), status, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            std::filesystem::remove(temp_blob_path);
            return 1;
        }
        blob_path = temp_blob_path.string();
        std::fprintf(stderr, "[pack] %s\n", status.c_str());
    }
    if (!mono27b_validate_blob_file(blob_path, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    Mono27BBlobHeader header{};
    if (!mono27b_read_blob_header(blob_path, header, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    Mono27BTokenizer tokenizer;
    if (!tokenizer.load_from_blob(blob_path, header, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    Mono27BBlobSection sections[4] = {};
    if (!mono27b_read_blob_sections(blob_path, sections, 4, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    Mono27BBlobSection weights_blob{};
    bool found_weights_blob = false;
    for (uint32_t i = 0; i < header.section_count && i < 4; ++i) {
        if (sections[i].kind == MONO27B_SECTION_WEIGHTS) {
            weights_blob = sections[i];
            found_weights_blob = true;
            break;
        }
    }
    if (!found_weights_blob) {
        std::fprintf(stderr, "missing weights section in blob\n");
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    Mono27BWeightEntry target_entry_s{}, draft_entry_s{}, head_entry_s{};
    if (!mono27b_scan_weight_entry_by_role(blob_path, MONO27B_ROLE_TARGET_MATVEC, target_entry_s, error)) {
        std::fprintf(stderr, "target embedding not found: %s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    if (!mono27b_scan_weight_entry_by_role(blob_path, MONO27B_ROLE_TARGET_OUTPUT_HEAD, head_entry_s, error)) {
        std::fprintf(stderr, "output head not found: %s\n", error.c_str());
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    if (!mono27b_scan_weight_entry_by_role(blob_path, MONO27B_ROLE_DRAFT_MATVEC, draft_entry_s, error)) {
        error.clear();
        if (!mono27b_scan_weight_entry_by_source_type(blob_path, MONO27B_SOURCE_DRAFT, MONO27B_GGML_TYPE_Q8_0, draft_entry_s, error)) {
            std::fprintf(stderr, "no draft tensor found: %s\n", error.c_str());
            if (!temp_blob_path.empty()) {
                std::filesystem::remove(temp_blob_path);
            }
            return 1;
        }
        draft_entry_s.role = MONO27B_ROLE_DRAFT_MATVEC;
    }
    const Mono27BWeightEntry * target_entry = &target_entry_s;
    const Mono27BWeightEntry * draft_entry = &draft_entry_s;
    const Mono27BWeightEntry * output_head_entry = &head_entry_s;

    std::vector<unsigned char> target_probe;
    std::vector<unsigned char> draft_tensor;
    std::vector<unsigned char> output_head_tensor;
    {
        std::FILE * fp = std::fopen(blob_path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "missing blob: %s\n", blob_path.c_str());
            if (!temp_blob_path.empty()) {
                std::filesystem::remove(temp_blob_path);
            }
            return 1;
        }
        target_probe.resize(static_cast<size_t>(target_entry->data_bytes));
        draft_tensor.resize(static_cast<size_t>(draft_entry->data_bytes));
        output_head_tensor.resize(static_cast<size_t>(output_head_entry->data_bytes));
        struct ReadReq {
            const Mono27BWeightEntry * entry;
            unsigned char * dst;
        };
        const ReadReq reqs[3] = {
            {target_entry, target_probe.data()},
            {draft_entry, draft_tensor.data()},
            {output_head_entry, output_head_tensor.data()},
        };
        for (const ReadReq & req : reqs) {
            const uint64_t off = weights_blob.offset + req.entry->data_offset;
            const size_t nbytes = static_cast<size_t>(req.entry->data_bytes);
            if (std::fseek(fp, static_cast<long>(off), SEEK_SET) != 0 ||
                std::fread(req.dst, 1, nbytes, fp) != nbytes) {
                std::fclose(fp);
                std::fprintf(stderr, "failed to read weight payload from weights section\n");
                if (!temp_blob_path.empty()) {
                    std::filesystem::remove(temp_blob_path);
                }
                return 1;
            }
        }
        std::fclose(fp);
    }
    Mono27BRuntimeLayout layout{};
    char exec_error[512] = {};
    if (!mono27b_executor_init(&header, args.max_ctx, &layout, exec_error, sizeof(exec_error))) {
        std::fprintf(stderr, "%s\n", exec_error);
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    std::fprintf(stderr,
                 "[state] kv=%zu rollback=%zu workspace=%zu bos=%u eos=%u vocab=%u merges=%u\n",
                 layout.kv_bytes, layout.rollback_bytes, layout.workspace_bytes,
                 tokenizer.bos_id(), tokenizer.eos_id(),
                 tokenizer.vocab_size(), tokenizer.merges_count());

    // Upload all weights to GPU once — warm GPU memory before decode loop
    std::fprintf(stderr, "[load] uploading weights to GPU (%.1f MB)...\n",
                 static_cast<double>(target_probe.size() + draft_tensor.size() + output_head_tensor.size()) / (1024.0 * 1024.0));

    Mono27BExecutorWeights gpu_weights{};
    if (!mono27b_executor_load_weights(
            target_probe.data(), target_probe.size(),
            target_entry->row_elems, target_entry->row_count,
            draft_tensor.data(), draft_tensor.size(),
            draft_entry->row_elems, draft_entry->row_count,
            output_head_tensor.data(), output_head_tensor.size(),
            output_head_entry->row_elems, output_head_entry->row_count,
            &gpu_weights,
            exec_error, sizeof(exec_error))) {
        std::fprintf(stderr, "failed to load weights to GPU: %s\n", exec_error);
        if (!temp_blob_path.empty()) {
            std::filesystem::remove(temp_blob_path);
        }
        return 1;
    }
    std::fprintf(stderr, "[load] GPU weights ready (target=%u×%u draft=%u×%u head=%u×%u)\n",
                 target_entry->row_count, target_entry->row_elems,
                 draft_entry->row_count, draft_entry->row_elems,
                 output_head_entry->row_count, output_head_entry->row_elems);

    // Load output_norm.weight (F32 scale for RMS norm before the LM head)
    {
        std::FILE * nfp = std::fopen(blob_path.c_str(), "rb");
        if (nfp) {
            Mono27BWeightsSection wsec{};
            bool loaded_norm = false;
            if (std::fseek(nfp, static_cast<long>(weights_blob.offset), SEEK_SET) == 0 &&
                std::fread(&wsec, sizeof(wsec), 1, nfp) == 1 &&
                wsec.magic == MONO27B_WEIGHTS_MAGIC) {
                const uint64_t entries_off = weights_blob.offset + sizeof(Mono27BWeightsSection);
                std::fseek(nfp, static_cast<long>(entries_off), SEEK_SET);
                for (uint32_t ei = 0; ei < wsec.entry_count && !loaded_norm; ++ei) {
                    Mono27BWeightEntry e{};
                    if (std::fread(&e, sizeof(e), 1, nfp) != 1) { break; }
                    if (e.ggml_type == 0U && e.source_model == MONO27B_SOURCE_TARGET &&
                        e.row_count == 1U && e.row_elems == target_entry->row_elems &&
                        e.data_bytes == static_cast<uint64_t>(e.row_elems) * sizeof(float) &&
                        strstr(e.name, "output_norm")) {
                        const uint64_t abs_off = weights_blob.offset + e.data_offset;
                        std::vector<float> ns(e.row_elems);
                        if (std::fseek(nfp, static_cast<long>(abs_off), SEEK_SET) == 0 &&
                            std::fread(ns.data(), sizeof(float), e.row_elems, nfp) == e.row_elems) {
                            if (mono27b_executor_set_norm_scale(&gpu_weights, ns.data(), e.row_elems,
                                                                exec_error, sizeof(exec_error))) {
                                std::fprintf(stderr, "[load] output_norm.weight (%u elems) loaded\n", e.row_elems);
                                loaded_norm = true;
                            } else {
                                std::fprintf(stderr, "[warn] norm_scale upload: %s\n", exec_error);
                            }
                        }
                    }
                }
            }
            std::fclose(nfp);
            if (!loaded_norm) {
                std::fprintf(stderr, "[load] output_norm.weight not found, using unit scale\n");
            }
        }
    }

    // Release host copies to free RAM — weights are now on GPU
    target_probe.clear();
    target_probe.shrink_to_fit();
    draft_tensor.clear();
    draft_tensor.shrink_to_fit();
    output_head_tensor.clear();
    output_head_tensor.shrink_to_fit();

    const std::vector<int32_t> prompt_ids = tokenizer.encode(args.prompt);
    std::fprintf(stderr, "[prompt] tokens=%zu\n", prompt_ids.size());
    std::vector<int32_t> context_ids = prompt_ids;
    std::vector<int32_t> decoded_ids;
    decoded_ids.reserve(static_cast<size_t>(args.max_gen > 0 ? args.max_gen : 1));

    for (int step = 0; step < std::max(1, args.max_gen); ++step) {
        char diag[256] = {};
        int32_t candidate_ids[32] = {};
        size_t candidate_count = 0;
        if (!mono27b_executor_run_step(
                &gpu_weights,
                context_ids.data(), context_ids.size(),
                candidate_ids, 32, &candidate_count,
                diag, sizeof(diag),
                exec_error, sizeof(exec_error))) {
            std::fprintf(stderr, "step error: %s\n", exec_error);
            mono27b_executor_free_weights(&gpu_weights);
            if (!temp_blob_path.empty()) {
                std::filesystem::remove(temp_blob_path);
            }
            return 1;
        }
        std::fprintf(stderr, "[step %d] %s\n", step, diag);

        int32_t chosen = -1;
        for (size_t i = 0; i < candidate_count; ++i) {
            const int32_t candidate = candidate_ids[i];
            if (tokenizer.is_terminal(candidate)) {
                chosen = candidate;
                break;
            }
            const std::string token_text = tokenizer.decode_one(candidate);
            if (!token_text.empty()) {
                chosen = candidate;
                break;
            }
        }
        if (chosen < 0 && candidate_count > 0) {
            chosen = candidate_ids[0];
        }
        if (chosen < 0) {
            break;
        }
        decoded_ids.push_back(chosen);
        context_ids.push_back(chosen);
        if (tokenizer.is_terminal(chosen)) {
            break;
        }
    }

    mono27b_executor_free_weights(&gpu_weights);

    const std::string decoded = tokenizer.decode(decoded_ids);
    if (decoded.empty() || !looks_readable(decoded)) {
        std::fprintf(stderr, "[warning] decoded text is not fully readable\n");
    }
    std::fprintf(stderr, "[generated] %s\n", sanitize_text(decoded).c_str());
    std::fprintf(stderr, "[assistant]\n%s\n", decoded.c_str());
    if (!temp_blob_path.empty()) {
        std::filesystem::remove(temp_blob_path);
    }
    return 0;
}
