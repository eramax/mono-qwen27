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

// ─── Config ──────────────────────────────────────────────────────────────────

struct SamplingConfig {
    int   top_k       = 40;
    float top_p       = 0.95f;
    float min_p       = 0.05f;
    float temperature = 0.8f;
    std::vector<std::string> banned_tokens = {"", "<|im_end|>", "<|fim_pad|>",
                                               "<|repo_name|>", "<|file_sep|>", "<|vision_pad|>"};
};

struct ChatTemplateConfig {
    std::string prefix = "<|im_start|>user\n";
    std::string suffix = "<|im_end|>\n<|im_start|>assistant\n༚\n";
};

struct Mono27BConfig {
    SamplingConfig         sampling;
    Mono27BKernelConfig    kernels;
    ChatTemplateConfig     chat_template;
};

static float json_get_float(const std::string & json, const std::string & key, float fallback) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    try { return std::stof(json.substr(pos + 1)); } catch (...) { return fallback; }
}

static int json_get_int(const std::string & json, const std::string & key, int fallback) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    try { return std::stoi(json.substr(pos + 1)); } catch (...) { return fallback; }
}

static std::string json_get_string(const std::string & json, const std::string & key, const std::string & fallback) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    auto q1 = json.find('"', pos + 1);
    if (q1 == std::string::npos) return fallback;
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return fallback;
    std::string val = json.substr(q1 + 1, q2 - q1 - 1);
    std::string out;
    out.reserve(val.size());
    for (size_t i = 0; i < val.size(); ++i) {
        if (val[i] == '\\' && i + 1 < val.size()) {
            if (val[i + 1] == 'n') { out += '\n'; ++i; continue; }
            if (val[i + 1] == 't') { out += '\t'; ++i; continue; }
            if (val[i + 1] == '\\') { out += '\\'; ++i; continue; }
        }
        out += val[i];
    }
    return out;
}

static std::vector<std::string> json_get_string_array(const std::string & json, const std::string & key) {
    std::vector<std::string> result;
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return result;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;
    size_t depth = 0;
    for (size_t i = pos; i < json.size(); ++i) {
        if (json[i] == '[') { depth++; continue; }
        if (json[i] == ']') { depth--; if (depth == 0) break; continue; }
        if (json[i] == '"') {
            auto q2 = json.find('"', i + 1);
            if (q2 != std::string::npos) {
                result.push_back(json.substr(i + 1, q2 - i - 1));
                i = q2;
            }
        }
    }
    return result;
}

static std::string json_extract_section(const std::string & json, const std::string & key) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) return "{}";
    pos = json.find('{', pos);
    if (pos == std::string::npos) return "{}";
    size_t depth = 0;
    for (size_t i = pos; i < json.size(); ++i) {
        if (json[i] == '{') depth++;
        if (json[i] == '}') { depth--; if (depth == 0) return json.substr(pos, i - pos + 1); }
    }
    return "{}";
}

static Mono27BConfig load_config(const std::string & path) {
    Mono27BConfig cfg;
    std::ifstream f(path);
    if (!f) return cfg;
    std::string content((std::istreambuf_iterator<char>(f)), {});

    auto sam = json_extract_section(content, "sampling");
    cfg.sampling.top_k       = json_get_int(sam, "top_k", cfg.sampling.top_k);
    cfg.sampling.top_p       = json_get_float(sam, "top_p", cfg.sampling.top_p);
    cfg.sampling.min_p       = json_get_float(sam, "min_p", cfg.sampling.min_p);
    cfg.sampling.temperature = json_get_float(sam, "temperature", cfg.sampling.temperature);
    auto banned = json_get_string_array(sam, "banned_tokens");
    if (!banned.empty()) cfg.sampling.banned_tokens = std::move(banned);

    auto ker = json_extract_section(content, "kernels");
    cfg.kernels.matvec_threads        = json_get_int(ker, "matvec_threads", cfg.kernels.matvec_threads);
    cfg.kernels.q4k_q8_threads        = json_get_int(ker, "q4k_q8_threads", cfg.kernels.q4k_q8_threads);
    cfg.kernels.q4k_q8_warp_count     = json_get_int(ker, "q4k_q8_warp_count", cfg.kernels.q4k_q8_warp_count);
    cfg.kernels.q4k_q8_smem_per_warp  = json_get_int(ker, "q4k_q8_smem_per_warp", cfg.kernels.q4k_q8_smem_per_warp);
    cfg.kernels.q6k_mt_threads        = json_get_int(ker, "q6k_mt_threads", cfg.kernels.q6k_mt_threads);
    cfg.kernels.elementwise_threads   = json_get_int(ker, "elementwise_threads", cfg.kernels.elementwise_threads);
    cfg.kernels.rms_norm_threads      = json_get_int(ker, "rms_norm_threads", cfg.kernels.rms_norm_threads);
    cfg.kernels.quant_threads         = json_get_int(ker, "quant_threads", cfg.kernels.quant_threads);
    cfg.kernels.argmax_threads        = json_get_int(ker, "argmax_threads", cfg.kernels.argmax_threads);
    cfg.kernels.lm_head_chunk_rows    = json_get_int(ker, "lm_head_chunk_rows", cfg.kernels.lm_head_chunk_rows);
    cfg.kernels.q8_scratch_max_blocks = json_get_int(ker, "q8_scratch_max_blocks", cfg.kernels.q8_scratch_max_blocks);
    cfg.kernels.q8_dp4a_fallback      = json_get_int(ker, "q8_dp4a_fallback_blocks", cfg.kernels.q8_dp4a_fallback);
    cfg.kernels.rms_eps               = json_get_float(ker, "rms_eps", cfg.kernels.rms_eps);
    cfg.kernels.rope_theta            = json_get_float(ker, "rope_theta", cfg.kernels.rope_theta);

    auto tmpl = json_extract_section(content, "chat_template");
    cfg.chat_template.prefix = json_get_string(tmpl, "system_prefix", cfg.chat_template.prefix);
    cfg.chat_template.suffix = json_get_string(tmpl, "system_suffix", cfg.chat_template.suffix);

    return cfg;
}

// ─── Tokenizer bootstrap ─────────────────────────────────────────────────────

static bool load_tokenizer(const Mono27BGgufFile & gguf, int max_ctx,
                            Mono27BTokenizer & tok, std::string & error) {
    char tmp_name[] = "/tmp/mono27b-tok-XXXXXX";
    int tmp_fd = mkstemp(tmp_name);
    if (tmp_fd < 0) { error = "mkstemp failed"; return false; }

    Mono27BTokenizerSection ts{};
    ts.magic       = MONO27B_TOKENIZER_MAGIC;
    ts.version     = 1U;
    ts.vocab_size  = gguf.metadata.vocab_size;
    ts.merges_count= gguf.metadata.merges_count;
    ts.bos_id      = gguf.metadata.bos_id;
    ts.eos_id      = gguf.metadata.eos_id;
    ts.im_start_id = gguf.metadata.im_start_id;
    ts.im_end_id   = gguf.metadata.im_end_id;

    std::vector<unsigned char> payload;
    auto push_str = [&](const std::string & s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        payload.insert(payload.end(), (unsigned char *)&len, (unsigned char *)&len + 4);
        payload.insert(payload.end(), s.begin(), s.end());
    };
    for (auto & t : gguf.metadata.tokens)  push_str(t);
    for (auto & m : gguf.metadata.merges)  push_str(m);

    Mono27BBlobHeader bh{};
    bh.file_bytes        = sizeof(bh) + 2 * sizeof(Mono27BBlobSection) + sizeof(ts) + payload.size();
    bh.section_count     = 2;
    bh.section_table_offset = sizeof(bh);
    bh.tokenizer_offset  = sizeof(bh) + 2 * sizeof(Mono27BBlobSection);
    bh.tokenizer_bytes   = sizeof(ts) + payload.size();
    bh.max_ctx_hint      = static_cast<uint32_t>(max_ctx);
    bh.target_layers     = MONO27B_TARGET_LAYERS;
    bh.draft_layers      = 5;
    std::snprintf(bh.target_arch,  sizeof(bh.target_arch),  "%s", MONO27B_TARGET_ARCH);
    std::snprintf(bh.draft_arch,   sizeof(bh.draft_arch),   "%s", MONO27B_DRAFT_ARCH);
    std::snprintf(bh.target_quant, sizeof(bh.target_quant), "%s", MONO27B_TARGET_QUANT);
    std::snprintf(bh.draft_quant,  sizeof(bh.draft_quant),  "%s", MONO27B_DRAFT_QUANT);

    Mono27BBlobSection secs[2] = {
        {MONO27B_SECTION_TOKENIZER, 0, bh.tokenizer_offset, bh.tokenizer_bytes},
        {MONO27B_SECTION_WEIGHTS,   0, 0, 0}
    };
    std::FILE * fp = ::fdopen(tmp_fd, "wb");
    std::fwrite(&bh,   sizeof(bh),   1, fp);
    std::fwrite(secs,  sizeof(secs),  1, fp);
    std::fwrite(&ts,   sizeof(ts),   1, fp);
    std::fwrite(payload.data(), 1, payload.size(), fp);
    std::fclose(fp);

    bool ok = tok.load_from_blob(tmp_name, bh, error);
    std::filesystem::remove(tmp_name);
    return ok;
}

// ─── Sampling ────────────────────────────────────────────────────────────────

static int sample(std::vector<float> logits, const Mono27BGgufFile & gguf,
                  std::mt19937 & rng, const SamplingConfig & cfg) {
    const int n = static_cast<int>(logits.size());
    for (int i = 0; i < n && i < (int)gguf.metadata.tokens.size(); ++i)
        for (auto & b : cfg.banned_tokens)
            if (gguf.metadata.tokens[i] == b) { logits[i] = -std::numeric_limits<float>::infinity(); break; }

    int best = 0;
    for (int i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;

    if (cfg.temperature <= 0.f) return best;

    int top_k = std::clamp(cfg.top_k, 1, n);
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    auto cmp = [&](int a, int b) { return logits[a] > logits[b]; };
    if (top_k < n) std::nth_element(order.begin(), order.begin() + top_k, order.end(), cmp);
    order.resize(top_k);
    std::sort(order.begin(), order.end(), cmp);

    float max_l = -std::numeric_limits<float>::infinity();
    for (int id : order) max_l = std::max(max_l, logits[id]);

    struct Cand { int id; float p; };
    std::vector<Cand> cands;
    float sum = 0.f;
    for (int id : order) {
        float p = std::exp((logits[id] - max_l) / cfg.temperature);
        cands.push_back({id, p});
        sum += p;
    }
    if (sum <= 0.f || !std::isfinite(sum)) return best;
    for (auto & c : cands) c.p /= sum;

    const float min_p_thresh = cands.front().p * cfg.min_p;
    cands.erase(std::remove_if(cands.begin(), cands.end(),
                               [&](const Cand & c) { return c.p < min_p_thresh; }),
                cands.end());
    if (cands.empty()) cands.push_back({best, 1.f});

    if (cfg.top_p < 1.f) {
        float acc = 0.f;
        size_t cut = cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            acc += cands[i].p;
            if (acc >= cfg.top_p) { cut = i + 1; break; }
        }
        cands.resize(cut);
    }

    float total = 0.f;
    for (auto & c : cands) total += c.p;
    if (total <= 0.f) return best;

    std::uniform_real_distribution<float> dist(0.f, total);
    float pick = dist(rng), run = 0.f;
    for (auto & c : cands) { run += c.p; if (pick <= run) return c.id; }
    return cands.back().id;
}

// ─── Performance stats ───────────────────────────────────────────────────────

static void print_stats(const std::string & model_path, int pos, int max_ctx,
                         int n_prompt, int n_gen,
                         double prefill_ms, double gen_ms) {
    auto tps = [](int n, double ms) { return n > 0 && ms > 0 ? n / ms * 1000.0 : 0.0; };

    size_t vram_free = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free, &vram_total);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    auto mb = [](size_t b) { return static_cast<long long>(b / (1024 * 1024)); };

    std::string name = model_path.substr(model_path.rfind('/') + 1);
    if (name.size() > 39) name.resize(39);

    char b1[64], b2[64], b3[64], b4[64], b5[64];
    std::snprintf(b1, sizeof(b1), "%.1f ms / %d tok = %.1f tok/s", prefill_ms, n_prompt, tps(n_prompt, prefill_ms));
    std::snprintf(b2, sizeof(b2), "%.1f ms / %d tok = %.1f tok/s", gen_ms,     n_gen,    tps(n_gen,    gen_ms));
    std::snprintf(b3, sizeof(b3), "%d / %d", pos, max_ctx);
    std::snprintf(b4, sizeof(b4), "%lld MiB / %lld MiB (%.1f%%)",
                  mb(vram_total - vram_free), mb(vram_total),
                  vram_total > 0 ? 100.0 * (vram_total - vram_free) / vram_total : 0.0);
    std::snprintf(b5, sizeof(b5), "%lld MiB", mb(vram_free));

    std::fprintf(stderr, "\n╔════════════════════════════════════════════════════╗\n");
    std::fprintf(stderr,   "║              Performance Summary                   ║\n");
    std::fprintf(stderr,   "╠════════════════════════════════════════════════════╣\n");
    std::fprintf(stderr,   "║  GPU       %-40s║\n", prop.name);
    std::fprintf(stderr,   "║  Model     %-40s║\n", name.c_str());
    std::fprintf(stderr,   "╠════════════════════════════════════════════════════╣\n");
    std::fprintf(stderr,   "║  Prefill   %-40s║\n", b1);
    std::fprintf(stderr,   "║  Generate  %-40s║\n", b2);
    std::fprintf(stderr,   "╠════════════════════════════════════════════════════╣\n");
    std::fprintf(stderr,   "║  Ctx       %-40s║\n", b3);
    std::fprintf(stderr,   "║  Output    %-40d tokens                       ║\n", n_gen);
    std::fprintf(stderr,   "║  VRAM      %-40s║\n", b4);
    std::fprintf(stderr,   "║  VRAM free %-40s║\n", b5);
    std::fprintf(stderr,   "╚════════════════════════════════════════════════════╝\n\n");
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
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
        std::fprintf(stderr, "gguf error: %s\n", gguf_err.c_str());
        return 1;
    }
    if (args.verbose && !gguf.metadata.chat_template.empty())
        std::fprintf(stderr, "[template] %s\n", gguf.metadata.chat_template.substr(0, 200).c_str());

    if (gguf.metadata.architecture != MONO27B_TARGET_ARCH ||
        gguf.metadata.embedding_length != MONO27B_TARGET_HIDDEN ||
        gguf.metadata.block_count != MONO27B_TARGET_LAYERS ||
        gguf.metadata.feed_forward_length != MONO27B_TARGET_FFN) {
        std::fprintf(stderr, "model mismatch (wrong arch/hparams)\n");
        return 1;
    }

    // mmap model file
    int fd = ::open(args.model.c_str(), O_RDONLY);
    if (fd < 0) { std::fprintf(stderr, "open failed: %s\n", args.model.c_str()); return 1; }
    struct stat st; ::fstat(fd, &st);
    size_t fsize = static_cast<size_t>(st.st_size);
    void * mdata = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mdata == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); return 1; }
    const unsigned char * data = static_cast<const unsigned char *>(mdata);

    // Tokenizer
    Mono27BTokenizer tokenizer;
    {
        std::string tok_err;
        if (!load_tokenizer(gguf, args.max_ctx, tokenizer, tok_err)) {
            std::fprintf(stderr, "tokenizer error: %s\n", tok_err.c_str());
            ::munmap(mdata, fsize);
            return 1;
        }
    }
    if (args.verbose)
        std::fprintf(stderr, "[state] vocab=%u merges=%u bos=%u eos=%u\n",
                     tokenizer.vocab_size(), tokenizer.merges_count(),
                     tokenizer.bos_id(), tokenizer.eos_id());

    // GPU state and weights
    Mono27BExecutorState state{};
    char err[512] = {};
    if (!mono27b_engine_init_state(args.max_ctx, &state, err, sizeof(err))) {
        std::fprintf(stderr, "state init error: %s\n", err);
        ::munmap(mdata, fsize);
        return 1;
    }
    state.max_ctx = args.max_ctx;

    Mono27BExecutorWeights weights{};
    if (!mono27b_engine_load_weights(data, gguf.data_offset,
                                      gguf.tensors.data(), gguf.tensors.size(),
                                      &weights, err, sizeof(err))) {
        std::fprintf(stderr, "weight load error: %s\n", err);
        mono27b_engine_free_state(&state);
        ::munmap(mdata, fsize);
        return 1;
    }
    cudaGetLastError();
    if (args.verbose) std::fprintf(stderr, "[load] GPU weights ready\n");

    // Prompt
    std::string prompt = args.prompt;
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' ||
                                prompt.back() == '\t' || prompt.back() == ' '))
        prompt.pop_back();
    if (args.chat)
        prompt = cfg.chat_template.prefix + prompt + cfg.chat_template.suffix;

    std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
    if (gguf.metadata.add_bos_token)
        prompt_ids.insert(prompt_ids.begin(), static_cast<int32_t>(tokenizer.bos_id()));

    if (args.verbose) {
        std::fprintf(stderr, "[prompt] tokens=%zu ids=", prompt_ids.size());
        for (int id : prompt_ids) std::fprintf(stderr, "%d ", id);
        std::fprintf(stderr, "\n");
    }

    // Inference pipeline
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now(), t1 = t0, t2 = t0, t3 = t0;
    std::vector<int32_t> generated;
    std::vector<float> last_logits;
    int pos = 0;

    auto decode = [&](int tok, int p) -> std::vector<float> {
        Mono27BLogitsOutput out{};
        if (!mono27b_engine_decode_step(&weights, &state, tok, p, &out, err, sizeof(err))) {
            std::fprintf(stderr, "decode error at pos %d: %s\n", p, err);
            return {};
        }
        std::vector<float> logits(MONO27B_TARGET_VOCAB);
        cudaMemcpy(logits.data(), out.logits, MONO27B_TARGET_VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
        mono27b_engine_free_logits(&out);
        return logits;
    };

    // Prefill
    t0 = clock::now();
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        auto logits = decode(prompt_ids[i], static_cast<int>(i));
        if (logits.empty()) goto cleanup;
        if (args.verbose) {
            int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::fprintf(stderr, "[prompt %zu] top1=%d max=%.4f\n", i, best, logits[best]);
        }
        if (i + 1 == prompt_ids.size()) last_logits = std::move(logits);
    }
    t1 = clock::now();

    // Generate
    pos = static_cast<int>(prompt_ids.size());
    t2 = clock::now();
    for (int step = 0; step < std::max(1, args.max_gen); ++step) {
        std::vector<float> * lp = &last_logits;
        std::vector<float> cur;
        if (step > 0 || last_logits.empty()) {
            cur = decode(generated.empty() ? prompt_ids.back() : generated.back(), pos++);
            if (cur.empty()) goto cleanup;
            lp = &cur;
        }

        int chosen = sample(*lp, gguf, rng, cfg.sampling);
        if (args.verbose) {
            int best = static_cast<int>(std::max_element(lp->begin(), lp->end()) - lp->begin());
            std::fprintf(stderr, "[step %d] chosen=%d top1=%d max=%.4f\n",
                         step, chosen, best, (*lp)[best]);
        } else {
            std::string piece = tokenizer.decode({chosen});
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        }

        if (tokenizer.is_terminal(chosen)) break;
        generated.push_back(chosen);
    }
    t3 = clock::now();

    std::fwrite("\n", 1, 1, stdout);
    std::fflush(stdout);

    if (args.verbose && !generated.empty())
        std::fprintf(stderr, "[output] %s\n", tokenizer.decode(generated).c_str());

    print_stats(args.model, pos, args.max_ctx,
                static_cast<int>(prompt_ids.size()), static_cast<int>(generated.size()),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t3 - t2).count());

#ifdef MONO27B_TIMING
    mono27b_engine_print_timing(&state);
#endif

cleanup:
    mono27b_engine_free_state(&state);
    mono27b_engine_free_weights(&weights);
    ::munmap(mdata, fsize);
    return 0;
}
