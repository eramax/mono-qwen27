#include "mono27b_cli.h"
#include "mono27b_config.h"
#include "mono27b_format.h"
#include "mono27b_gguf.h"
#include "mono27b_tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
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

static bool is_banned_token(const std::string & token) {
    return token == "<|endoftext|>" ||
           token == "<|im_end|>" ||
           token == "<|fim_pad|>" ||
           token == "<|repo_name|>" ||
           token == "<|file_sep|>" ||
           token == "<|vision_pad|>";
}

static std::vector<int32_t> load_replay_tokens(const std::string & path) {
    std::vector<int32_t> tokens;
    if (path.empty()) return tokens;

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "warn: failed to open replay trace: %s\n", path.c_str());
        return tokens;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line.rfind("phase\t", 0) == 0) continue;
        size_t p0 = 0;
        size_t p1 = line.find('\t', p0);
        if (p1 == std::string::npos) continue;
        const std::string phase = line.substr(p0, p1 - p0);
        if (phase != "gen") continue;
        p0 = p1 + 1;
        p1 = line.find('\t', p0);
        if (p1 == std::string::npos) continue;
        p0 = p1 + 1;
        p1 = line.find('\t', p0);
        if (p1 == std::string::npos) continue;
        p0 = p1 + 1;
        p1 = line.find('\t', p0);
        if (p1 == std::string::npos) continue;
        p0 = p1 + 1;
        p1 = line.find('\t', p0);
        if (p1 == std::string::npos) continue;
        try {
            tokens.push_back(static_cast<int32_t>(std::stol(line.substr(p0, p1 - p0))));
        } catch (...) {
            continue;
        }
    }

    std::fprintf(stderr, "[replay] loaded %zu generation tokens from %s\n",
                 tokens.size(), path.c_str());
    return tokens;
}

static int sample_from_logits(std::vector<float> logits, const Mono27BGgufFile & gguf,
                              std::mt19937 & rng, int top_k, float top_p, float min_p,
                              float temperature, int * best_out) {
    const int n_vocab = static_cast<int>(logits.size());
    if (n_vocab == 0) return 0;

    for (size_t i = 0; i < gguf.metadata.tokens.size() && i < logits.size(); ++i) {
        if (is_banned_token(gguf.metadata.tokens[i])) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }

    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    if (best_out) *best_out = best;

    top_k = std::clamp(top_k, 1, n_vocab);
    std::vector<int> order(n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[i] = i;
    auto cmp = [&](int a, int b) { return logits[a] > logits[b]; };
    if (top_k < n_vocab) {
        std::nth_element(order.begin(), order.begin() + top_k, order.end(), cmp);
    }
    order.resize(top_k);
    std::sort(order.begin(), order.end(), cmp);

    if (temperature <= 0.0f) {
        return best;
    }

    const float temp = temperature;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int id : order) max_logit = std::max(max_logit, logits[id]);

    struct Cand { int id; float p; };
    std::vector<Cand> cands;
    cands.reserve(order.size());
    float sum = 0.0f;
    for (int id : order) {
        const float z = (logits[id] - max_logit) / temp;
        const float p = std::exp(z);
        cands.push_back({id, p});
        sum += p;
    }
    if (sum <= 0.0f || !std::isfinite(sum)) return best;
    for (auto & c : cands) c.p /= sum;

    const float best_p = cands.front().p;
    std::vector<Cand> filtered;
    filtered.reserve(cands.size());
    for (const auto & c : cands) {
        if (c.p < best_p * min_p) continue;
        filtered.push_back(c);
    }
    if (filtered.empty()) filtered = cands;

    if (top_p < 1.0f) {
        std::vector<Cand> nucleus;
        nucleus.reserve(filtered.size());
        float nuc_sum = 0.0f;
        for (const auto & c : filtered) {
            nucleus.push_back(c);
            nuc_sum += c.p;
            if (nuc_sum >= top_p) break;
        }
        if (!nucleus.empty()) filtered.swap(nucleus);
    }

    float total = 0.0f;
    for (const auto & c : filtered) total += c.p;
    if (total <= 0.0f || !std::isfinite(total)) return best;

    std::uniform_real_distribution<float> dist(0.0f, total);
    float pick = dist(rng);
    float running = 0.0f;
    for (const auto & c : filtered) {
        running += c.p;
        if (pick <= running) return c.id;
    }
    return filtered.back().id;
}

static int argmax_from_logits(const std::vector<float> & logits, int * best_out = nullptr) {
    const int n_vocab = static_cast<int>(logits.size());
    if (n_vocab == 0) return 0;
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    if (best_out) *best_out = best;
    return best;
}

int main(int argc, char ** argv) {
    Mono27BChatArgs args;
    if (!mono27b_parse_chat_args(argc, argv, args) || args.show_help) {
        mono27b_print_chat_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }
    const bool quiet = args.quiet;
    std::mt19937 rng(args.seed);

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
    if (!quiet && !gguf.metadata.chat_template.empty()) {
        std::fprintf(stderr, "[template] %s\n", gguf.metadata.chat_template.substr(0, 200).c_str());
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
    const int top_k = gguf.metadata.has_sampling_top_k
        ? static_cast<int>(gguf.metadata.sampling_top_k)
        : 40;
    const float top_p = gguf.metadata.has_sampling_top_p
        ? gguf.metadata.sampling_top_p
        : 0.95f;
    const float min_p = 0.05f;
    const float temperature = gguf.metadata.has_sampling_temp
        ? gguf.metadata.sampling_temp
        : 0.8f;

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

    if (!quiet) {
        std::fprintf(stderr, "[state] vocab=%u merges=%u bos=%u eos=%u\n",
                     tokenizer.vocab_size(), tokenizer.merges_count(),
                     tokenizer.bos_id(), tokenizer.eos_id());
    }

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
    if (!quiet) std::fprintf(stderr, "[load] GPU weights ready\n");

    // Raw completion mode: match llama.cpp --no-cnv behavior for plain prompts.
    std::string user_prompt = args.prompt;
    while (!user_prompt.empty() && (user_prompt.back() == '\n' || user_prompt.back() == '\r' ||
                                    user_prompt.back() == '\t' || user_prompt.back() == ' ')) {
        user_prompt.pop_back();
    }

    // --chat: render the same ChatML framing used by llama.cpp conversation mode.
    if (args.chat) {
        user_prompt = "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    std::vector<int32_t> prompt_ids;
    prompt_ids = tokenizer.encode(user_prompt);
    if (gguf.metadata.add_bos_token) {
        prompt_ids.insert(prompt_ids.begin(), static_cast<int32_t>(tokenizer.bos_id()));
    }
    if (!quiet) {
        std::fprintf(stderr, "[prompt] tokens=%zu ids=", prompt_ids.size());
        for (size_t i = 0; i < prompt_ids.size(); ++i) fprintf(stderr, "%d ", prompt_ids[i]);
        fprintf(stderr, "\n");
    }

    char errbuf[512] = {};
    int pos = 0;
    int last_tok = 0;
    std::vector<int32_t> generated;
    std::vector<float> last_prompt_logits;
    using clock = std::chrono::steady_clock;
    auto t_prefill_start = clock::now();
    auto t_prefill_end = t_prefill_start;
    auto t_gen_start = t_prefill_start;
    auto t_gen_end = t_prefill_start;
    std::ofstream trace;
    std::FILE * debug_fp = nullptr;
    std::vector<int32_t> replay_tokens = load_replay_tokens(args.replay_trace_path);
    if (!args.trace_path.empty()) {
        trace.open(args.trace_path, std::ios::out | std::ios::trunc);
        if (!trace) {
            std::fprintf(stderr, "error: failed to open trace file: %s\n", args.trace_path.c_str());
            goto cleanup;
        }
        trace << "phase\tstep\tpos\tinput_id\tchosen_id\tbest_id\tbest_logit\ttop_ids\ttop_logits\n";
    }
    if (!args.debug_path.empty()) {
        debug_fp = std::fopen(args.debug_path.c_str(), "w");
        if (!debug_fp) {
            std::fprintf(stderr, "error: failed to open debug file: %s\n", args.debug_path.c_str());
            goto cleanup;
        }
        std::fprintf(debug_fp, "phase\tstep\tpos\ttok\tlabel\tn\tmin\tmax\tmean\tl2\tvalues\n");
    }
    t_prefill_start = clock::now();
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        Mono27BLogitsOutput logits{};
        if (!mono27b_engine_decode_step(&gpu_weights, &state,
                                         prompt_ids[i], static_cast<int>(i),
                                         &logits, debug_fp, args.debug_pos, errbuf, sizeof(errbuf))) {
            std::fprintf(stderr, "step error at prompt %zu: %s\n", i, errbuf);
            mono27b_engine_free_logits(&logits);
            goto cleanup;
        }
        std::vector<float> logits_host(MONO27B_TARGET_VOCAB);
        cudaError_t copy_err = cudaMemcpy(logits_host.data(), logits.logits,
                                          MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
        if (copy_err != cudaSuccess) {
            std::fprintf(stderr, "step %zu logits copy error: %s\n", i, cudaGetErrorString(copy_err));
        }
        int best = 0;
        int chosen = prompt_ids[i];
        best = argmax_from_logits(logits_host, &best);
        // Save full logits for verification (like llama-debug --save-logits)
        {
            std::string logit_path = "/tmp/our_logits_" + std::to_string(i) + ".bin";
            FILE *lf = std::fopen(logit_path.c_str(), "wb");
            if (lf) {
                std::fwrite(logits_host.data(), sizeof(float), MONO27B_TARGET_VOCAB, lf);
                std::fclose(lf);
            }
        }
        if (trace) {
            write_trace_row(trace, "prompt", static_cast<int>(i), static_cast<int>(i),
                            prompt_ids[i], chosen, logits_host);
            trace.flush();
        }
        if (!quiet) std::fprintf(stderr, "[prompt %zu] top1=%d max=%.4f\n", i, best, logits_host[best]);
        (void)best;
        if (i == prompt_ids.size() - 1) {
            last_prompt_logits = std::move(logits_host);
        }
        mono27b_engine_free_logits(&logits);
    }
    t_prefill_end = clock::now();

    // Generate tokens — step 0 reuses logits from last prompt token (matching llama.cpp).
    pos = static_cast<int>(prompt_ids.size());
    last_tok = prompt_ids.empty() ? 0 : prompt_ids.back();

    t_gen_start = clock::now();

    for (int step = 0; step < std::max(1, args.max_gen); ++step) {
        std::vector<float> logits_host;

        if (step == 0 && !last_prompt_logits.empty()) {
            logits_host = std::move(last_prompt_logits);
        } else {
            Mono27BLogitsOutput logits{};
            if (!mono27b_engine_decode_step(&gpu_weights, &state,
                                             last_tok, pos,
                                             &logits, debug_fp, args.debug_pos, errbuf, sizeof(errbuf))) {
                std::fprintf(stderr, "step error at gen %d: %s\n", step, errbuf);
                mono27b_engine_free_logits(&logits);
                goto cleanup;
            }
            logits_host.resize(MONO27B_TARGET_VOCAB);
            cudaError_t copy_err = cudaMemcpy(logits_host.data(), logits.logits,
                                              MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
            if (copy_err != cudaSuccess) {
                std::fprintf(stderr, "step %d logits copy error: %s\n", step, cudaGetErrorString(copy_err));
            }
            mono27b_engine_free_logits(&logits);
            pos++;
        }

        int best = 0;
        int chosen = -1;
        if (step < static_cast<int>(replay_tokens.size())) {
            chosen = replay_tokens[static_cast<size_t>(step)];
            best = argmax_from_logits(logits_host, &best);
            std::fprintf(stderr, "[replay] step %d forcing token %d\n", step, chosen);
        } else if (args.greedy) {
            chosen = argmax_from_logits(logits_host, &best);
        } else {
            chosen = sample_from_logits(logits_host, gguf, rng, top_k, top_p, min_p, temperature, &best);
        }

        if (trace) {
            write_trace_row(trace, "gen", step, pos, last_tok, chosen, logits_host);
            trace.flush();
        }

        if (!quiet) std::fprintf(stderr, "[step %d] top1=%d max=%.4f chosen=%d\n", step, best, logits_host[best], chosen);

        if (quiet) {
            std::string piece = tokenizer.decode({chosen});
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        }

        if (tokenizer.is_terminal(chosen)) break;

        generated.push_back(chosen);
        last_tok = chosen;
    }
    t_gen_end = clock::now();

    // Decode and output
    if (quiet && !generated.empty()) {
        // In quiet mode tokens were streamed live; just add a final newline
        std::fwrite("\n", 1, 1, stdout);
        std::fflush(stdout);
    } else if (!generated.empty()) {
        std::string text = tokenizer.decode(generated);
        std::fprintf(stderr, "[generated] %s\n", sanitize_text(text).c_str());
        std::fprintf(stderr, "[assistant]\n%s\n", text.c_str());
    } else if (!quiet) {
        std::fprintf(stderr, "[generated] (no tokens)\n");
    }

    // Print performance stats
    {
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
        int n_prompt = static_cast<int>(prompt_ids.size());
        int n_gen = static_cast<int>(generated.size());
        double prefill_tps = (n_prompt > 0 && prefill_ms > 0) ? (n_prompt / prefill_ms * 1000.0) : 0.0;
        double gen_tps = (n_gen > 0 && gen_ms > 0) ? (n_gen / gen_ms * 1000.0) : 0.0;

        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        size_t vram_used = vram_total - vram_free;

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);

        auto mb = [](size_t bytes) -> long long { return static_cast<long long>(bytes / (1024 * 1024)); };

        // Box width = 54 chars. Content area = 39 chars (between label end and right border)
        // Extract model filename from path
        std::string model_name = target_path;
        auto last_slash = model_name.rfind('/');
        if (last_slash != std::string::npos) model_name = model_name.substr(last_slash + 1);
        if (model_name.size() > 39) model_name.resize(39);

        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔════════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║              Performance Summary                   ║\n");
        std::fprintf(stderr, "╠════════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  GPU       %-40s║\n", prop.name);
        std::fprintf(stderr, "║  Model     %-40s║\n", model_name.c_str());
        std::fprintf(stderr, "╠════════════════════════════════════════════════════╣\n");

        // Timing rows
        char buf1[64], buf2[64];
        std::snprintf(buf1, sizeof(buf1), "%.1f ms / %d tok = %.1f tok/s", prefill_ms, n_prompt, prefill_tps);
        std::snprintf(buf2, sizeof(buf2), "%.1f ms / %d tok = %.1f tok/s", gen_ms, n_gen, gen_tps);
        std::fprintf(stderr, "║  Prefill   %-40s║\n", buf1);
        std::fprintf(stderr, "║  Generate  %-40s║\n", buf2);

        std::fprintf(stderr, "╠════════════════════════════════════════════════════╣\n");

        // Context + tokens
        char buf3[64], buf4[64], buf5[64];
        std::snprintf(buf3, sizeof(buf3), "%d / %d", pos, args.max_ctx);
        std::snprintf(buf4, sizeof(buf4), "%d tokens", n_gen);
        std::snprintf(buf5, sizeof(buf5), "%lld MiB / %lld MiB (%.1f%%)",
                      mb(vram_used), mb(vram_total),
                      vram_total > 0 ? (100.0 * vram_used / vram_total) : 0.0);
        std::fprintf(stderr, "║  Ctx       %-40s║\n", buf3);
        std::fprintf(stderr, "║  Output    %-40s║\n", buf4);
        std::fprintf(stderr, "║  VRAM      %-40s║\n", buf5);

        char buf6[32];
        std::snprintf(buf6, sizeof(buf6), "%lld MiB", mb(vram_free));
        std::fprintf(stderr, "║  VRAM free %-40s║\n", buf6);

        std::fprintf(stderr, "╚════════════════════════════════════════════════════╝\n");
        std::fprintf(stderr, "\n");
    }

cleanup:
    if (debug_fp) std::fclose(debug_fp);
    mono27b_engine_free_state(&state);
    mono27b_engine_free_weights(&gpu_weights);
    munmap(mmap_data, file_size);
    return 0;
}
