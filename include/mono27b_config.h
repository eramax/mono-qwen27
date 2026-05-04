#pragma once

#include "mono27b_format.h"

#include <cstddef>
#include <cstdio>

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

// M-RoPE: Qwen3.5 uses sections [text=11, height=11, width=10, extra=0] pairs = 64 total dims.
// For text-only input, only the text section uses the token position.
constexpr int MONO27B_N_ROT_DIMS = 64;
constexpr int MONO27B_N_ROT_DIMS_S0 = 22; // section 0: 11 pairs * 2 = 22 dims

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
    void * q8_scratch;  // Q8_1 buffer for matvec (544 blocks × 36 bytes)
    int max_ctx;
    int kv_len;
};

struct Mono27BLogitsOutput {
    float * logits;
};

extern "C" bool mono27b_executor_init(const Mono27BBlobHeader * header,
                                       int max_ctx,
                                       Mono27BRuntimeLayout * layout,
                                       char * error, size_t error_cap);

extern "C" bool mono27b_executor_load_weights(
    const unsigned char * target_host, size_t target_bytes,
    uint32_t target_row_elems, uint32_t target_row_count,
    const unsigned char * draft_host, size_t draft_bytes,
    uint32_t draft_row_elems, uint32_t draft_row_count,
    const unsigned char * head_host, size_t head_bytes,
    uint32_t head_row_elems, uint32_t head_row_count,
    Mono27BExecutorWeights * out,
    char * error, size_t error_cap);

extern "C" void mono27b_executor_free_weights(Mono27BExecutorWeights * w);

extern "C" bool mono27b_executor_set_norm_scale(
    Mono27BExecutorWeights * w,
    const float * scale_host, uint32_t elems,
    char * error, size_t error_cap);

extern "C" bool mono27b_executor_run_step(
    const Mono27BExecutorWeights * w,
    const int32_t * token_ids, size_t token_count,
    int32_t * generated_ids, size_t generated_cap,
    size_t * generated_count,
    char * diag, size_t diag_cap,
    char * error, size_t error_cap);

extern "C" bool mono27b_executor_run_prompt(
    const char * prompt,
    const int32_t * prompt_ids, size_t prompt_id_count,
    const unsigned char * target_tensor, size_t target_tensor_bytes,
    uint32_t target_row_elems, uint32_t target_row_count,
    const unsigned char * draft_tensor, size_t draft_tensor_bytes,
    uint32_t draft_row_elems, uint32_t draft_row_count,
    const unsigned char * output_head_tensor, size_t output_head_tensor_bytes,
    uint32_t output_head_row_elems, uint32_t output_head_row_count,
    int32_t * generated_ids, size_t generated_cap,
    size_t * generated_count,
    char * output, size_t output_cap,
    char * error, size_t error_cap);

// New engine API
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
    FILE * debug_fp,
    int debug_pos,
    char * error, size_t error_cap);

extern "C" void mono27b_engine_free_logits(Mono27BLogitsOutput * output);
