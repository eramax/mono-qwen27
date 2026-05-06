#include "mono27b_cli.h"
#include "mono27b_config.h"
#include "mono27b_format.h"
#include "mono27b_gguf.h"
#include "mono27b_tokenizer.h"

#include <algorithm>
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

bool g_mono27b_verbose = false;
Mono27BKernelConfig g_kernel_cfg;

struct SamplingConfig {
    int top_k;
    float top_p;
    float min_p;
    float temperature;
    std::vector<std::string> banned_tokens;
};

struct ChatTemplateConfig {
    std::string prefix;       // tokenized as part of input prompt
    std::string suffix;       // tokenized as part of input prompt (control tokens)
    std::string visible_tail; // appended to suffix in tokenization AND echoed to stdout
                              // so the user sees the assistant prefix (e.g. "<think>\n")
};

struct Mono27BConfig {
    SamplingConfig sampling;
    Mono27BKernelConfig kernels;
    ChatTemplateConfig chat_template;
};

template<typename T>
static T read_json_value(const std::string &s, size_t pos);

template<> inline int read_json_value<int>(const std::string &s, size_t pos) {
    while (pos < s.size() && std::isspace(s[pos])) pos++;
    return std::stoi(s.substr(pos));
}

template<> inline float read_json_value<float>(const std::string &s, size_t pos) {
    while (pos < s.size() && std::isspace(s[pos])) pos++;
    return std::stof(s.substr(pos));
}

template<> inline std::string read_json_value<std::string>(const std::string &s, size_t pos) {
    while (pos < s.size() && std::isspace(s[pos])) pos++;
    if (pos >= s.size() || s[pos] != '"') return "";
    size_t end = s.find('"', pos + 1);
    if (end == std::string::npos) return "";
    std::string val = s.substr(pos + 1, end - pos - 1);
    std::string r;
    r.reserve(val.size());
    for (size_t i = 0; i < val.size(); i++) {
        if (val[i] == '\\' && i + 1 < val.size()) {
            if (val[i+1] == 'n') { r += '\n'; i++; continue; }
            if (val[i+1] == 't') { r += '\t'; i++; continue; }
            if (val[i+1] == '\\') { r += '\\'; i++; continue; }
        }
        r += val[i];
    }
    return r;
}

template<typename T>
static T get_json_value(const std::string &json, const std::string &key, T fallback) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    try { return read_json_value<T>(json, pos + 1); }
    catch (...) { return fallback; }
}

static Mono27BConfig load_config(const std::string &path) {
    Mono27BConfig cfg;
    std::ifstream f(path);
    if (!f) return cfg;
    std::string json((std::istreambuf_iterator<char>(f)), {});

    cfg.sampling.top_k = get_json_value(json, "top_k", 40);
    cfg.sampling.top_p = get_json_value(json, "top_p", 0.95f);
    cfg.sampling.min_p = get_json_value(json, "min_p", 0.05f);
    cfg.sampling.temperature = get_json_value(json, "temperature", 0.8f);
    cfg.chat_template.prefix = get_json_value(json, "system_prefix", std::string("<|im_start|>user\n"));
    cfg.chat_template.suffix = get_json_value(json, "system_suffix", std::string("\n<|im_end|>\n<|im_start|>assistant"));
    cfg.chat_template.visible_tail = get_json_value(json, "assistant_visible_tail", std::string(""));

    auto samp_pos = json.find("\"sampling\"");
    auto banned_pos = (samp_pos != std::string::npos) ? json.find("banned_tokens", samp_pos) : std::string::npos;
    if (banned_pos != std::string::npos) {
        auto arr_pos = json.find('[', banned_pos);
        if (arr_pos != std::string::npos) {
            auto end_pos = json.find(']', arr_pos);
            if (end_pos != std::string::npos) {
                std::string arr_str = json.substr(arr_pos + 1, end_pos - arr_pos - 1);
                for (size_t i = 0; i < arr_str.size(); i++) {
                    if (arr_str[i] == '"') {
                        auto j = arr_str.find('"', i + 1);
                        if (j != std::string::npos) {
                            cfg.sampling.banned_tokens.push_back(arr_str.substr(i + 1, j - i - 1));
                            i = j;
                        }
                    }
                }
            }
        }
    }

    cfg.kernels.matvec_threads = get_json_value(json, "matvec_threads", 128);
    cfg.kernels.q4k_q8_threads = get_json_value(json, "q4k_q8_threads", 128);
    cfg.kernels.q4k_q8_warp_count = get_json_value(json, "q4k_q8_warp_count", 4);
    cfg.kernels.q4k_q8_smem_per_warp = get_json_value(json, "q4k_q8_smem_per_warp", 8);
    cfg.kernels.q6k_mt_threads = get_json_value(json, "q6k_mt_threads", 256);
    cfg.kernels.elementwise_threads = get_json_value(json, "elementwise_threads", 256);
    cfg.kernels.rms_norm_threads = get_json_value(json, "rms_norm_threads", 256);
    cfg.kernels.quant_threads = get_json_value(json, "quant_threads", 128);
    cfg.kernels.argmax_threads = get_json_value(json, "argmax_threads", 512);
    cfg.kernels.lm_head_chunk_rows = get_json_value(json, "lm_head_chunk_rows", 4096);
    cfg.kernels.q8_scratch_max_blocks = get_json_value(json, "q8_scratch_max_blocks", 2048);
    cfg.kernels.q8_dp4a_fallback = get_json_value(json, "q8_dp4a_fallback_blocks", 544);
    cfg.kernels.rms_eps = get_json_value(json, "rms_eps", 1e-6f);
    cfg.kernels.rope_theta = get_json_value(json, "rope_theta", 10000000.0f);

    return cfg;
}

static bool load_tokenizer(const Mono27BGgufFile &gguf, int max_ctx,
                            Mono27BTokenizer &tok, std::string &error) {
    char tmp_name[] = "/tmp/mono27b-tok-XXXXXX";
    int tmp_fd = mkstemp(tmp_name);
    if (tmp_fd < 0) {
        error = "mkstemp failed";
        return false;
    }

    Mono27BTokenizerSection ts{};
    ts.magic = MONO27B_TOKENIZER_MAGIC;
    ts.version = 1U;
    ts.vocab_size = gguf.metadata.vocab_size;
    ts.merges_count = gguf.metadata.merges_count;
    ts.bos_id = gguf.metadata.bos_id;
    ts.eos_id = gguf.metadata.eos_id;
    ts.im_start_id = gguf.metadata.im_start_id;
    ts.im_end_id = gguf.metadata.im_end_id;

    std::vector<unsigned char> payload;
    auto push_str = [&](const std::string &s) {
        uint32_t len = s.size();
        payload.insert(payload.end(), (unsigned char *)&len, (unsigned char *)&len + 4);
        payload.insert(payload.end(), s.begin(), s.end());
    };
    for (auto &t : gguf.metadata.tokens) push_str(t);
    for (auto &m : gguf.metadata.merges) push_str(m);

    Mono27BBlobHeader bh{};
    bh.file_bytes = sizeof(bh) + 2 * sizeof(Mono27BBlobSection) + sizeof(ts) + payload.size();
    bh.section_count = 2;
    bh.section_table_offset = sizeof(bh);
    bh.tokenizer_offset = sizeof(bh) + 2 * sizeof(Mono27BBlobSection);
    bh.tokenizer_bytes = sizeof(ts) + payload.size();
    bh.max_ctx_hint = max_ctx;
    bh.target_layers = MONO27B_TARGET_LAYERS;
    bh.draft_layers = 5;
    snprintf(bh.target_arch, sizeof(bh.target_arch), "%s", MONO27B_TARGET_ARCH);
    snprintf(bh.draft_arch, sizeof(bh.draft_arch), "%s", MONO27B_DRAFT_ARCH);
    snprintf(bh.target_quant, sizeof(bh.target_quant), "%s", MONO27B_TARGET_QUANT);
    snprintf(bh.draft_quant, sizeof(bh.draft_quant), "%s", MONO27B_DRAFT_QUANT);

    Mono27BBlobSection secs[2] = {
        {MONO27B_SECTION_TOKENIZER, 0, bh.tokenizer_offset, bh.tokenizer_bytes},
        {MONO27B_SECTION_WEIGHTS, 0, 0, 0}
    };

    FILE *fp = fdopen(tmp_fd, "wb");
    fwrite(&bh, sizeof(bh), 1, fp);
    fwrite(secs, sizeof(secs), 1, fp);
    fwrite(&ts, sizeof(ts), 1, fp);
    fwrite(payload.data(), 1, payload.size(), fp);
    fclose(fp);

    bool ok = tok.load_from_blob(tmp_name, bh, error);
    std::filesystem::remove(tmp_name);
    return ok;
}

static int sample(std::vector<float> logits, const Mono27BGgufFile &gguf,
                  std::mt19937 &rng, const SamplingConfig &cfg) {
    int n = logits.size();

    for (int i = 0; i < n && i < (int)gguf.metadata.tokens.size(); i++)
        for (auto &b : cfg.banned_tokens)
            if (gguf.metadata.tokens[i] == b) {
                logits[i] = -std::numeric_limits<float>::infinity();
                break;
            }

    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;

    if (cfg.temperature <= 0.f) return best;

    int top_k = std::clamp(cfg.top_k, 1, n);
    std::vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = i;

    auto cmp = [&](int a, int b) { return logits[a] > logits[b]; };
    if (top_k < n) std::nth_element(ids.begin(), ids.begin() + top_k, ids.end(), cmp);
    ids.resize(top_k);
    std::sort(ids.begin(), ids.end(), cmp);

    float max_l = logits[ids[0]];
    std::vector<std::pair<int, float>> cands;
    float sum = 0.f;

    for (int id : ids) {
        float p = std::exp((logits[id] - max_l) / cfg.temperature);
        cands.push_back({id, p});
        sum += p;
    }

    if (sum <= 0.f || !std::isfinite(sum)) return best;

    for (auto &c : cands) c.second /= sum;

    float min_p_thresh = cands[0].second * cfg.min_p;
    auto it = std::remove_if(cands.begin(), cands.end(),
                            [min_p_thresh](auto &c) { return c.second < min_p_thresh; });
    cands.erase(it, cands.end());
    if (cands.empty()) cands.push_back({best, 1.f});

    if (cfg.top_p < 1.f) {
        float acc = 0.f;
        for (size_t i = 0; i < cands.size(); i++) {
            acc += cands[i].second;
            if (acc >= cfg.top_p) {
                cands.resize(i + 1);
                break;
            }
        }
    }

    float total = 0.f;
    for (auto &c : cands) total += c.second;

    std::uniform_real_distribution<float> dist(0.f, total);
    float pick = dist(rng), run = 0.f;
    for (auto &c : cands) {
        run += c.second;
        if (pick <= run) return c.first;
    }
    return cands.back().first;
}

static void print_stats(const std::string &model_path, int pos, int max_ctx,
                        int n_prompt, int n_gen, double prefill_ms, double gen_ms) {
    auto tps = [](int n, double ms) { return n > 0 && ms > 0 ? n / ms * 1000.0 : 0.0; };

    size_t vram_free = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free, &vram_total);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::string name = model_path.substr(model_path.rfind('/') + 1);
    if (name.size() > 39) name.resize(39);

    fprintf(stderr, "\n─ Stats ─────────────────────────────────────\n");
    fprintf(stderr, "GPU: %s\n", prop.name);
    fprintf(stderr, "Model: %s\n", name.c_str());
    fprintf(stderr, "Prefill: %.1f ms / %d tok = %.1f tok/s\n", prefill_ms, n_prompt, tps(n_prompt, prefill_ms));
    fprintf(stderr, "Gen: %.1f ms / %d tok = %.1f tok/s\n", gen_ms, n_gen, tps(n_gen, gen_ms));
    fprintf(stderr, "Ctx: %d / %d\n", pos, max_ctx);
    fprintf(stderr, "VRAM: %zu / %zu MiB\n", (vram_total - vram_free) / (1024 * 1024), vram_total / (1024 * 1024));
    fprintf(stderr, "───────────────────────────────────────────\n\n");
}

int main(int argc, char **argv) {
    Mono27BChatArgs args;
    if (!mono27b_parse_args(argc, argv, args) || args.show_help) {
        mono27b_print_usage(argv[0]);
        return args.show_help ? 0 : 1;
    }
    g_mono27b_verbose = args.verbose;

    const Mono27BConfig cfg = load_config("config.json");
    g_kernel_cfg = cfg.kernels;
    std::mt19937 rng(args.seed);

    cudaDeviceReset();
    cudaSetDevice(0);

    Mono27BGgufFile gguf;
    std::string gguf_err;
    if (!mono27b_read_gguf(args.model, gguf, gguf_err)) {
        fprintf(stderr, "gguf error: %s\n", gguf_err.c_str());
        return 1;
    }

    if (gguf.metadata.architecture != MONO27B_TARGET_ARCH ||
        gguf.metadata.embedding_length != MONO27B_TARGET_HIDDEN ||
        gguf.metadata.block_count != MONO27B_TARGET_LAYERS ||
        gguf.metadata.feed_forward_length != MONO27B_TARGET_FFN) {
        fprintf(stderr, "model mismatch\n");
        return 1;
    }

    int fd = open(args.model.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open failed: %s\n", args.model.c_str());
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    size_t fsize = st.st_size;
    void *mdata = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mdata == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }
    const unsigned char *data = (const unsigned char *)mdata;

    Mono27BTokenizer tokenizer;
    {
        std::string tok_err;
        if (!load_tokenizer(gguf, args.max_ctx, tokenizer, tok_err)) {
            fprintf(stderr, "tokenizer error: %s\n", tok_err.c_str());
            munmap(mdata, fsize);
            return 1;
        }
    }

    Mono27BExecutorState state{};
    char err[512] = {};
    if (!mono27b_engine_init_state(args.max_ctx, &state, err, sizeof(err))) {
        fprintf(stderr, "state init error: %s\n", err);
        munmap(mdata, fsize);
        return 1;
    }
    state.max_ctx = args.max_ctx;

    Mono27BExecutorWeights weights{};
    if (!mono27b_engine_load_weights(data, gguf.data_offset,
                                      gguf.tensors.data(), gguf.tensors.size(),
                                      &weights, err, sizeof(err))) {
        fprintf(stderr, "weight load error: %s\n", err);
        mono27b_engine_free_state(&state);
        munmap(mdata, fsize);
        return 1;
    }
    cudaGetLastError();

    std::string prompt = args.prompt;
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' ||
                                prompt.back() == '\t' || prompt.back() == ' '))
        prompt.pop_back();
    prompt = cfg.chat_template.prefix + prompt + cfg.chat_template.suffix + cfg.chat_template.visible_tail;

    std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
    if (gguf.metadata.add_bos_token)
        prompt_ids.insert(prompt_ids.begin(), tokenizer.bos_id());

    using clock = std::chrono::steady_clock;
    auto decode = [&](int tok, int p) -> std::vector<float> {
        Mono27BLogitsOutput out{};
        if (!mono27b_engine_decode_step(&weights, &state, tok, p, &out, err, sizeof(err))) {
            fprintf(stderr, "decode error at pos %d: %s\n", p, err);
            return {};
        }
        std::vector<float> logits(MONO27B_TARGET_VOCAB);
        cudaMemcpy(logits.data(), out.logits, MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
        mono27b_engine_free_logits(&out);
        return logits;
    };

    bool ok = true;
    auto t0 = clock::now(), t1 = t0, t2 = t0, t3 = t0;
    std::vector<float> last_logits;
    std::vector<int32_t> generated;
    int pos = prompt_ids.size();

    for (size_t i = 0; i < prompt_ids.size(); i++) {
        auto logits = decode(prompt_ids[i], i);
        if (logits.empty()) { ok = false; break; }
        if (i + 1 == prompt_ids.size()) last_logits = logits;
    }
    t1 = clock::now();

    if (ok) {
        if (!cfg.chat_template.visible_tail.empty()) {
            fwrite(cfg.chat_template.visible_tail.data(), 1,
                   cfg.chat_template.visible_tail.size(), stdout);
            fflush(stdout);
        }
        t2 = clock::now();
        for (int step = 0; step < std::max(1, args.max_gen); step++) {
            auto step_start = clock::now();
            double decode_ms = 0.0;
            int cur_pos = pos - 1;
            std::vector<float> cur;
            auto *lp = &last_logits;
            if (step > 0 || last_logits.empty()) {
                auto decode_start = clock::now();
                cur = decode(generated.empty() ? prompt_ids.back() : generated.back(), pos++);
                auto decode_end = clock::now();
                decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
                if (cur.empty()) { ok = false; break; }
                lp = &cur;
                cur_pos = pos - 1;
            }

            int chosen = sample(*lp, gguf, rng, cfg.sampling);
            if (args.verbose) {
                int best = std::max_element(lp->begin(), lp->end()) - lp->begin();
                fprintf(stderr, "[step %d] chosen=%d top1=%d max=%.4f\n", step, chosen, best, (*lp)[best]);
            } else {
                std::string piece = tokenizer.decode({chosen});
                fwrite(piece.data(), 1, piece.size(), stdout);
                fflush(stdout);
            }
            auto step_end = clock::now();
            if (args.trace_gen) {
                double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
                double tok_s = step_ms > 0.0 ? 1000.0 / step_ms : 0.0;
                fprintf(stderr,
                        "[gen %d] pos=%d token=%d decode_ms=%.3f step_ms=%.3f tok/s=%.2f\n",
                        step, cur_pos, chosen, decode_ms, step_ms, tok_s);
            }

            if (tokenizer.is_terminal(chosen)) break;
            generated.push_back(chosen);
        }
        t3 = clock::now();
    }

    fwrite("\n", 1, 1, stdout);
    fflush(stdout);

    if (ok && !args.quiet) {
        print_stats(args.model, pos, args.max_ctx, prompt_ids.size(), generated.size(),
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

#ifdef MONO27B_TIMING
    mono27b_engine_print_timing(&state);
#endif

    mono27b_engine_free_state(&state);
    mono27b_engine_free_weights(&weights);
    munmap(mdata, fsize);
    return ok ? 0 : 1;
}
