#pragma once

#include "mono27b_format.h"

#include <cstddef>

constexpr int MONO27B_TARGET_HEAD_DIM = 256;
constexpr int MONO27B_TARGET_N_HEAD = 24;
constexpr int MONO27B_TARGET_N_KV_HEAD = 4;
constexpr int MONO27B_TARGET_FA_LAYERS = 16;
constexpr int MONO27B_TARGET_SSM_LAYERS = 48;
constexpr int MONO27B_DRAFT_BLOCK = 16;

struct Mono27BRuntimeLayout {
    size_t kv_bytes = 0;
    size_t rollback_bytes = 0;
    size_t workspace_bytes = 0;
    size_t state_bytes = 0;
};

// Pre-loaded GPU weights handle — upload once, decode many times
struct Mono27BExecutorWeights {
    void * target_gpu = nullptr;
    size_t target_bytes = 0;
    uint32_t target_row_elems = 0;
    uint32_t target_row_count = 0;

    void * draft_gpu = nullptr;
    size_t draft_bytes = 0;
    uint32_t draft_row_elems = 0;
    uint32_t draft_row_count = 0;

    void * head_gpu = nullptr;
    size_t head_bytes = 0;
    uint32_t head_row_elems = 0;
    uint32_t head_row_count = 0;

    // Optional: output RMS-norm scale weights (F32, hidden_dim elements)
    void * norm_scale_gpu = nullptr;
    uint32_t norm_scale_elems = 0;
};

extern "C" bool mono27b_executor_init(const Mono27BBlobHeader * header,
                                       int max_ctx,
                                       Mono27BRuntimeLayout * layout,
                                       char * error,
                                       size_t error_cap);

// Upload all weights to GPU once; returns a handle for fast decode
extern "C" bool mono27b_executor_load_weights(
    const unsigned char * target_host, size_t target_bytes,
    uint32_t target_row_elems, uint32_t target_row_count,
    const unsigned char * draft_host, size_t draft_bytes,
    uint32_t draft_row_elems, uint32_t draft_row_count,
    const unsigned char * head_host, size_t head_bytes,
    uint32_t head_row_elems, uint32_t head_row_count,
    Mono27BExecutorWeights * out,
    char * error, size_t error_cap);

// Free GPU weight buffers
extern "C" void mono27b_executor_free_weights(Mono27BExecutorWeights * w);

// Upload the output RMS-norm scale vector (F32) to GPU
// Call after load_weights; scale_host may be null to skip (norm is applied without scale)
extern "C" bool mono27b_executor_set_norm_scale(
    Mono27BExecutorWeights * w,
    const float * scale_host, uint32_t elems,
    char * error, size_t error_cap);

// Fast decode step using pre-loaded GPU weights
extern "C" bool mono27b_executor_run_step(
    const Mono27BExecutorWeights * w,
    const int32_t * token_ids, size_t token_count,
    int32_t * generated_ids, size_t generated_cap,
    size_t * generated_count,
    char * diag, size_t diag_cap,
    char * error, size_t error_cap);

extern "C" bool mono27b_executor_run_prompt(const char * prompt,
                                             const int32_t * prompt_ids,
                                             size_t prompt_id_count,
                                             const unsigned char * target_tensor,
                                             size_t target_tensor_bytes,
                                             uint32_t target_row_elems,
                                             uint32_t target_row_count,
                                             const unsigned char * draft_tensor,
                                             size_t draft_tensor_bytes,
                                             uint32_t draft_row_elems,
                                             uint32_t draft_row_count,
                                             const unsigned char * output_head_tensor,
                                             size_t output_head_tensor_bytes,
                                             uint32_t output_head_row_elems,
                                             uint32_t output_head_row_count,
                                             int32_t * generated_ids,
                                             size_t generated_cap,
                                             size_t * generated_count,
                                             char * output,
                                             size_t output_cap,
                                             char * error,
                                             size_t error_cap);
