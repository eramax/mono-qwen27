#include "mono27b_cli.h"
#include "mono27b_config.h"
#include "mono27b_format.h"
#include "mono27b_gguf.h"
#include "mono27b_tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

static std::string sanitize_text(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (unsigned char ch : in) {
        if (ch == '\n') { out += "\\n"; }
        else if (ch == '\r') { out += "\\r"; }
        else if (ch == '\t') { out += "\\t"; }
        else if (ch < 0x80 && std::isprint(ch)) { out += static_cast<char>(ch); }
        else { char buf[8]; std::snprintf(buf, sizeof(buf), "\\x%02x", ch); out += buf; }
    }
    return out;
}

static void write_trace_row(std::ostream & os, const char * phase, int step, int pos,
                            int input_id, int chosen_id, const std::vector<float> & logits) {
    struct Candidate { int id; float v; };
    Candidate best[8];
    int best_n = 0;
    float best_v = -1e30f;
    int best_id = -1;
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_id = i;
        }
        int ins = best_n;
        while (ins > 0 && v > best[ins - 1].v) --ins;
        if (ins < 8) {
            if (best_n < 8) ++best_n;
            for (int j = best_n - 1; j > ins; --j) best[j] = best[j - 1];
            best[ins] = {i, v};
        }
    }

    os << phase << '\t' << step << '\t' << pos << '\t' << input_id << '\t'
       << chosen_id << '\t' << best_id << '\t' << best_v << '\t';
    for (int i = 0; i < best_n; ++i) {
        if (i) os << ',';
        os << best[i].id;
    }
    os << '\t';
    for (int i = 0; i < best_n; ++i) {
        if (i) os << ',';
        os << best[i].v;
    }
    os << '\n';
}

int main(int argc, char ** argv) {
    Mono27BChatArgs args;
    if (!mono27b_parse_chat_args(argc, argv, args) || args.show_help) {
        mono27b_print_chat_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }

    // Determine model file path
    std::string target_path = args.target_gguf;
    if (target_path.empty()) {
        std::fprintf(stderr, "error: no target model specified (use -m <path>)\n");
        return 1;
    }

    cudaDeviceReset();
    cudaSetDevice(0);
    // Read GGUF info
    Mono27BGgufFile gguf;
    std::string gguf_error;
    if (!mono27b_read_gguf(target_path, gguf, gguf_error)) {
        std::fprintf(stderr, "gguf error: %s\n", gguf_error.c_str());
        return 1;
    }

    // Verify architecture
    if (gguf.metadata.architecture != MONO27B_TARGET_ARCH) {
        std::fprintf(stderr, "architecture mismatch: got %s, expected %s\n",
                     gguf.metadata.architecture.c_str(), MONO27B_TARGET_ARCH);
        return 1;
    }
    if (gguf.metadata.embedding_length != MONO27B_TARGET_HIDDEN ||
        gguf.metadata.block_count != MONO27B_TARGET_LAYERS ||
        gguf.metadata.feed_forward_length != MONO27B_TARGET_FFN) {
        std::fprintf(stderr, "hparams mismatch\n");
        return 1;
    }

    // mmap the GGUF file
    int fd = ::open(target_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "failed to open: %s\n", target_path.c_str());
        return 1;
    }
    struct stat st;
    ::fstat(fd, &st);
    size_t file_size = static_cast<size_t>(st.st_size);
    void * mmap_data = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mmap_data == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed: %s\n", target_path.c_str());
        return 1;
    }

    const unsigned char * data = static_cast<const unsigned char *>(mmap_data);

    // Initialize tokenizer from GGUF metadata
    Mono27BTokenizer tokenizer;
    {
        std::string tok_error;
        // Write a temp blob with the tokenizer data to reuse existing code
        char tmp_name[] = "/tmp/mono27b-tok-XXXXXX";
        int tmp_fd = mkstemp(tmp_name);
        if (tmp_fd < 0) {
            std::fprintf(stderr, "failed to create temp tokenizer file\n");
            munmap(mmap_data, file_size);
            return 1;
        }
        // Build the tokenizer blob section
        Mono27BTokenizerSection ts{};
        ts.magic = MONO27B_TOKENIZER_MAGIC;
        ts.version = 1U;
        ts.vocab_size = gguf.metadata.vocab_size;
        ts.merges_count = gguf.metadata.merges_count;
        ts.bos_id = gguf.metadata.bos_id;
        ts.eos_id = gguf.metadata.eos_id;
        ts.im_start_id = gguf.metadata.im_start_id;
        ts.im_end_id = gguf.metadata.im_end_id;

        // Write tokenizer section + vocab/merges to temp file
        std::vector<unsigned char> tok_bytes;
        auto append_u32 = [&](uint32_t v) {
            tok_bytes.insert(tok_bytes.end(), (unsigned char*)&v, (unsigned char*)&v + 4);
        };
        auto append_str = [&](const std::string & s) {
            uint32_t len = static_cast<uint32_t>(s.size());
            append_u32(len);
            tok_bytes.insert(tok_bytes.end(), s.begin(), s.end());
        };
        for (auto & t : gguf.metadata.tokens) append_str(t);
        for (auto & m : gguf.metadata.merges) append_str(m);

        // Write blob header
        Mono27BBlobHeader bh{};
        bh.file_bytes = sizeof(bh) + 2 * sizeof(Mono27BBlobSection) + sizeof(ts) + tok_bytes.size();
        bh.section_count = 2;
        bh.section_table_offset = sizeof(bh);
        bh.tokenizer_offset = sizeof(bh) + 2 * sizeof(Mono27BBlobSection);
        bh.tokenizer_bytes = sizeof(ts) + tok_bytes.size();
        bh.max_ctx_hint = static_cast<uint32_t>(args.max_ctx);
        bh.target_layers = MONO27B_TARGET_LAYERS;
        bh.draft_layers = 5;
        std::snprintf(bh.target_arch, sizeof(bh.target_arch), "%s", MONO27B_TARGET_ARCH);
        std::snprintf(bh.draft_arch, sizeof(bh.draft_arch), "%s", MONO27B_DRAFT_ARCH);
        std::snprintf(bh.target_quant, sizeof(bh.target_quant), "%s", MONO27B_TARGET_QUANT);
        std::snprintf(bh.draft_quant, sizeof(bh.draft_quant), "%s", MONO27B_DRAFT_QUANT);

        std::FILE * tmp_fp = ::fdopen(tmp_fd, "wb");
        std::fwrite(&bh, sizeof(bh), 1, tmp_fp);
        Mono27BBlobSection secs[2] = {
            {MONO27B_SECTION_TOKENIZER, 0, bh.tokenizer_offset, bh.tokenizer_bytes},
            {MONO27B_SECTION_WEIGHTS, 0, 0, 0}
        };
        std::fwrite(secs, sizeof(secs), 1, tmp_fp);
        std::fwrite(&ts, sizeof(ts), 1, tmp_fp);
        std::fwrite(tok_bytes.data(), 1, tok_bytes.size(), tmp_fp);
        std::fclose(tmp_fp);

        if (!tokenizer.load_from_blob(tmp_name, bh, tok_error)) {
            std::fprintf(stderr, "tokenizer error: %s\n", tok_error.c_str());
            std::filesystem::remove(tmp_name);
            munmap(mmap_data, file_size);
            return 1;
        }
        std::filesystem::remove(tmp_name);
    }

    std::fprintf(stderr, "[state] vocab=%u merges=%u bos=%u eos=%u\n",
                 tokenizer.vocab_size(), tokenizer.merges_count(),
                 tokenizer.bos_id(), tokenizer.eos_id());

    // Memory tracking
    // Initialize runtime state BEFORE loading weights to avoid fragmentation
    cudaDeviceSynchronize();
    Mono27BExecutorState state{};
    char state_err[512] = {};
    if (!mono27b_engine_init_state(args.max_ctx, &state, state_err, sizeof(state_err))) {
        std::fprintf(stderr, "state init error: %s\n", state_err);
        munmap(mmap_data, file_size);
        return 1;
    }
    state.max_ctx = args.max_ctx;

    // Load weights to GPU
    Mono27BExecutorWeights gpu_weights{};
    char load_err[1024] = {};
    if (!mono27b_engine_load_weights(data, gguf.data_offset,
                                      gguf.tensors.data(), gguf.tensors.size(),
                                      &gpu_weights, load_err, sizeof(load_err))) {
        std::fprintf(stderr, "weight load error: %s\n", load_err);
        mono27b_engine_free_state(&state);
        munmap(mmap_data, file_size);
        return 1;
    }
    cudaGetLastError();
    std::fprintf(stderr, "[load] GPU weights ready\n");

    // Apply chat template and encode
    std::string chat_prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
                            "<|im_start|>user\n" + args.prompt + "\n<|im_end|>\n"
                            "<|im_start|>assistant\n";
    std::vector<int32_t> prompt_ids = tokenizer.encode(chat_prompt);
    std::fprintf(stderr, "[prompt] tokens=%zu ids=", prompt_ids.size());
    for (size_t i = 0; i < prompt_ids.size(); ++i) fprintf(stderr, "%d ", prompt_ids[i]);
    fprintf(stderr, "\n");

    char errbuf[512] = {};
    int pos = 0;
    int last_tok = 0;
    std::vector<int32_t> generated;
    std::ofstream trace;
    if (!args.trace_path.empty()) {
        trace.open(args.trace_path, std::ios::out | std::ios::trunc);
        if (!trace) {
            std::fprintf(stderr, "error: failed to open trace file: %s\n", args.trace_path.c_str());
            goto cleanup;
        }
        trace << "phase\tstep\tpos\tinput_id\tchosen_id\tbest_id\tbest_logit\ttop_ids\ttop_logits\n";
    }

    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        Mono27BLogitsOutput logits{};
        if (!mono27b_engine_decode_step(&gpu_weights, &state,
                                         prompt_ids[i], static_cast<int>(i),
                                         &logits, errbuf, sizeof(errbuf))) {
            std::fprintf(stderr, "step error at prompt %zu: %s\n", i, errbuf);
            mono27b_engine_free_logits(&logits);
            goto cleanup;
        }
        std::vector<float> logits_host(MONO27B_TARGET_VOCAB);
        cudaMemcpy(logits_host.data(), logits.logits,
                   MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
        int best = 0;
        float best_v = logits_host[0];
        for (int j = 1; j < MONO27B_TARGET_VOCAB; ++j) {
            if (logits_host[j] > best_v) { best_v = logits_host[j]; best = j; }
        }
        if (trace) {
            write_trace_row(trace, "prompt", static_cast<int>(i), static_cast<int>(i),
                            prompt_ids[i], prompt_ids[i], logits_host);
            trace.flush();
        }
        std::fprintf(stderr, "[prompt %zu] top1=%d max=%.4f\n", i, best, best_v);
        mono27b_engine_free_logits(&logits);
    }

    // Generate tokens
    pos = static_cast<int>(prompt_ids.size());
    last_tok = prompt_ids.empty() ? 0 : prompt_ids.back();

    for (int step = 0; step < std::max(1, args.max_gen); ++step) {
        Mono27BLogitsOutput logits{};
        if (!mono27b_engine_decode_step(&gpu_weights, &state,
                                         last_tok, pos,
                                         &logits, errbuf, sizeof(errbuf))) {
            std::fprintf(stderr, "step error at gen %d: %s\n", step, errbuf);
            mono27b_engine_free_logits(&logits);
            goto cleanup;
        }

        // Download logits and find argmax
        std::vector<float> logits_host(MONO27B_TARGET_VOCAB);
        cudaMemcpy(logits_host.data(), logits.logits,
                   MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
        int best = 0;
        float best_v = logits_host[0];
        for (int j = 1; j < MONO27B_TARGET_VOCAB; ++j) {
            if (logits_host[j] > best_v) { best_v = logits_host[j]; best = j; }
        }
        int chosen = best;

        if (trace) {
            write_trace_row(trace, "gen", step, pos, last_tok, chosen, logits_host);
            trace.flush();
        }

        std::fprintf(stderr, "[step %d] top1=%d max=%.4f chosen=%d\n", step, best, best_v, chosen);
        mono27b_engine_free_logits(&logits);

        if (tokenizer.is_terminal(chosen)) break;

        generated.push_back(chosen);
        last_tok = chosen;
        pos++;
    }

    // Decode and output
    if (!generated.empty()) {
        std::string text = tokenizer.decode(generated);
        std::fprintf(stderr, "[generated] %s\n", sanitize_text(text).c_str());
        std::fprintf(stderr, "[assistant]\n%s\n", text.c_str());
    } else {
        std::fprintf(stderr, "[generated] (no tokens)\n");
    }

cleanup:
    mono27b_engine_free_state(&state);
    mono27b_engine_free_weights(&gpu_weights);
    munmap(mmap_data, file_size);
    return 0;
}
