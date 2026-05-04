#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_fp16_t;

typedef struct {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xE0 << 23;
    const float exp_scale = 0x1.0p-112f;
    union { uint32_t u; float f; } u;
    u.u = (two_w >> 4) + exp_offset;
    const float normalized_value = u.f * exp_scale;
    const uint32_t magic_mask = 126 << 23;
    const float magic_bias = 0.5f;
    u.u = (two_w >> 17) | magic_mask;
    const float denormalized_value = u.f - magic_bias;
    const uint32_t denormalized_cutoff = 1 << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? *(uint32_t*)&denormalized_value : *(uint32_t*)&normalized_value);
    u.u = result;
    return u.f;
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const block_q4_K * x, float * y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = ggml_compute_fp16_to_fp32(x[i].d);
        const float min = ggml_compute_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

int main() {
    block_q4_K qb;
    // ed 04 39 10 ef f4 a6 f1 e6 be b2 bf e8 ef cc e8
    // aa 97 db 7f 70 98 ea a3 9b 66 85 d3 94 d7 aa f6
    uint8_t bytes[] = {
        0xed, 0x04, 0x39, 0x10, 0xef, 0xf4, 0xa6, 0xf1, 0xe6, 0xbe, 0xb2, 0xbf, 0xe8, 0xef, 0xcc, 0xe8,
        0xaa, 0x97, 0xdb, 0x7f, 0x70, 0x98, 0xea, 0xa3, 0x9b, 0x66, 0x85, 0xd3, 0x94, 0xd7, 0xaa, 0xf6
    };
    memcpy(&qb, bytes, sizeof(bytes));
    float y[256];
    dequantize_row_q4_K(&qb, y, 256);
    for (int i = 0; i < 8; ++i) printf("%.10g ", y[i]);
    printf("\n");
    return 0;
}
