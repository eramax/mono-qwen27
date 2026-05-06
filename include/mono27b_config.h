#pragma once

#include "mono27b_format.h"

#include <cstddef>

extern bool g_mono27b_verbose;

// Runtime kernel tuning — set from config.json
struct Mono27BKernelConfig {
    int   matvec_threads        = 128;
    int   q4k_q8_threads        = 128;
    int   q4k_q8_warp_count     = 4;
    int   q4k_q8_smem_per_warp  = 8;
    int   q6k_mt_threads        = 256;
    int   elementwise_threads   = 256;
    int   rms_norm_threads      = 256;
    int   quant_threads         = 128;
    int   argmax_threads        = 512;
    int   lm_head_chunk_rows    = 4096;
    int   q8_scratch_max_blocks = 2048;
    int   q8_dp4a_fallback      = 544;
    float rms_eps               = 1e-6f;
    float rope_theta            = 10000000.0f;
};

extern Mono27BKernelConfig g_kernel_cfg;

// Model architecture constants — Qwen3.5-27B hybrid
constexpr int MONO27B_TARGET_N_HEAD     = 24;
constexpr int MONO27B_TARGET_N_KV_HEAD  = 4;
constexpr int MONO27B_TARGET_HEAD_DIM   = 256;
constexpr int MONO27B_TARGET_Q_DIM      = MONO27B_TARGET_N_HEAD * MONO27B_TARGET_HEAD_DIM;
constexpr int MONO27B_TARGET_KV_DIM     = MONO27B_TARGET_N_KV_HEAD * MONO27B_TARGET_HEAD_DIM;
constexpr int MONO27B_TARGET_VOCAB      = 248320;
constexpr int MONO27B_TARGET_FA_INTERVAL = 4;
constexpr float MONO27B_TARGET_ROPE_THETA = 10000000.0f;
constexpr float MONO27B_RMS_EPS         = 1e-6f;
constexpr int MONO27B_TARGET_FA_LAYERS  = MONO27B_TARGET_LAYERS / MONO27B_TARGET_FA_INTERVAL;

// SSM dimensions
constexpr int MONO27B_SSM_D_INNER   = 6144;
constexpr int MONO27B_SSM_D_STATE   = 128;
constexpr int MONO27B_SSM_DT_RANK   = 48;
constexpr int MONO27B_SSM_N_GROUP   = 16;
constexpr int MONO27B_SSM_CONV_KERN = 4;
constexpr int MONO27B_SSM_HEAD_V    = MONO27B_SSM_D_INNER / MONO27B_SSM_DT_RANK;
constexpr int MONO27B_SSM_HEAD_K    = MONO27B_SSM_D_STATE;
constexpr int MONO27B_SSM_CONV_CH   = MONO27B_SSM_D_INNER + 2 * MONO27B_SSM_N_GROUP * MONO27B_SSM_D_STATE;

// M-RoPE sections
constexpr int MONO27B_N_ROT_DIMS    = 64;
constexpr int MONO27B_N_ROT_DIMS_S0 = 22;

struct Mono27BRuntimeLayout {
    size_t kv_bytes = 0;
    size_t rollback_bytes = 0;
    size_t workspace_bytes = 0;
    size_t state_bytes = 0;
};

struct WeightView {
    void * ptr;
    uint32_t ggml_type;
    uint32_t row_blocks;
    uint32_t row_count;
};

struct Mono27BAttentionLayerWeights {
    WeightView attn_norm;
    WeightView wq;
    WeightView wk;
    WeightView wv;
    WeightView wo;
    WeightView gate;
    WeightView q_norm;
    WeightView k_norm;
    WeightView post_norm;
    WeightView ffn_gate;
    WeightView ffn_up;
    WeightView ffn_down;
};

struct Mono27BSSMLayerWeights {
    WeightView attn_norm;
    WeightView wqkv;
    WeightView wqkv_gate;
    WeightView ssm_conv1d;
    WeightView ssm_beta;
    WeightView ssm_alpha;
    WeightView ssm_dt_bias;
    WeightView ssm_a_log;
    WeightView ssm_norm;
    WeightView ssm_out;
    WeightView post_norm;
    WeightView ffn_gate;
    WeightView ffn_up;
    WeightView ffn_down;
};

struct Mono27BExecutorWeights {
    struct {
        bool is_attention;
        union {
            Mono27BAttentionLayerWeights attn;
            Mono27BSSMLayerWeights ssm;
        };
    } layers[MONO27B_TARGET_LAYERS];

    WeightView output_norm;
    WeightView lm_head;
    WeightView tok_embd;
};

struct Mono27BExecutorState {
    float * kv_cache_k[MONO27B_TARGET_FA_LAYERS];
    float * kv_cache_v[MONO27B_TARGET_FA_LAYERS];
    float * ssm_state[MONO27B_TARGET_LAYERS - MONO27B_TARGET_FA_LAYERS];
    float * conv_state[MONO27B_TARGET_LAYERS - MONO27B_TARGET_FA_LAYERS];
    float * work_buf;
    size_t work_buf_size;
    void * q8_scratch;
    int * argmax_result;
    void * cuda_graph;
    void * cuda_graph_exec;
    int graph_captured;
    int max_ctx;
    int kv_len;
    void * stream1;
    void * sync_event;

#ifdef MONO27B_TIMING
    static constexpr int MAX_TIMING_EVENTS = 65536;
    static constexpr int MAX_TIMING_LABELS = 64;
    void * timing_events[MAX_TIMING_EVENTS];
    int timing_event_count;
    int timing_event_capacity;
    const char * timing_labels[MAX_TIMING_EVENTS];
    float timing_acc_ms[MAX_TIMING_LABELS];
    int timing_acc_count[MAX_TIMING_LABELS];
    const char * timing_acc_label[MAX_TIMING_LABELS];
    int timing_acc_entries;
    int timing_tokens;
#endif
};

struct Mono27BLogitsOutput {
    float * logits;
};

extern "C" bool mono27b_engine_load_weights(
    const unsigned char * gguf_data, uint64_t data_offset,
    const struct Mono27BGgufTensorInfo * tensors, size_t tensor_count,
    Mono27BExecutorWeights * gpu_weights,
    char * error, size_t error_cap);

extern "C" void mono27b_engine_free_weights(Mono27BExecutorWeights * w);

extern "C" bool mono27b_engine_init_state(int max_ctx,
                                           Mono27BExecutorState * state,
                                           char * error, size_t error_cap);

extern "C" void mono27b_engine_free_state(Mono27BExecutorState * state);

extern "C" bool mono27b_engine_embed(
    const Mono27BExecutorWeights * we, int token_id,
    float * hidden, char * error, size_t error_cap);

extern "C" bool mono27b_engine_decode_step(
    const Mono27BExecutorWeights * weights,
    Mono27BExecutorState * state,
    int token_id, int position,
    Mono27BLogitsOutput * output,
    char * error, size_t error_cap);

extern "C" void mono27b_engine_free_logits(Mono27BLogitsOutput * output);

extern "C" int mono27b_engine_argmax(Mono27BExecutorState * st, const float * logits, int n);
extern "C" void mono27b_engine_print_timing(Mono27BExecutorState * st);
