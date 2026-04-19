/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define SEQ_LEN 32
#define EMBED_DIM 64
#define SOFTMAX_Q_BITS 7
#define SOFTMAX_ROW_SUM ((1 << SOFTMAX_Q_BITS) - 1)
#define SOFTMAX_SCORE_SCALE 8
#define SOFTMAX_BUCKET_STRIDE 16
#define SOFTMAX_LUT_LAST 16

// Equivalent GEMMs:
//   Scores = Q * K^T : M = 32, K = 64, N = 32
//   Output = A * V   : M = 32, K = 32, N = 64
//
// Softmax notes:
//   - scores are first scaled by sqrt(EMBED_DIM) ~= 8 using integer division
//   - the exponent uses a small lookup table in Q12
//   - each LUT bucket represents an extra 16 counts of post-scale score delta
//   - row probabilities are quantized to Q0.7 before the final A * V GEMM

static void softmax_rows_q7(int8_t probs[SEQ_LEN][SEQ_LEN],
                            int32_t scores[SEQ_LEN][SEQ_LEN]) {
  for (int i = 0; i < SEQ_LEN; ++i) {
    int32_t row_max_scaled = scores[i][0] / SOFTMAX_SCORE_SCALE;
    int best_j = 0;
    uint16_t row_weights[SEQ_LEN];
    uint32_t weight_sum = 0;

    for (int j = 1; j < SEQ_LEN; ++j) {
      int32_t scaled = scores[i][j] / SOFTMAX_SCORE_SCALE;
      if (scaled > row_max_scaled) {
        row_max_scaled = scaled;
        best_j = j;
      }
    }

    for (int j = 0; j < SEQ_LEN; ++j) {
      int32_t scaled = scores[i][j] / SOFTMAX_SCORE_SCALE;
      int32_t delta = row_max_scaled - scaled;
      if (delta < 0)
        delta = 0;
      int bucket = (delta + SOFTMAX_BUCKET_STRIDE - 1) / SOFTMAX_BUCKET_STRIDE;
      if (bucket > SOFTMAX_LUT_LAST)
        bucket = SOFTMAX_LUT_LAST;
      uint16_t weight = 1;
      if (bucket <= 0)
        weight = 4096;
      if (bucket == 1)
        weight = 2484;
      if (bucket == 2)
        weight = 1507;
      if (bucket == 3)
        weight = 914;
      if (bucket == 4)
        weight = 554;
      if (bucket == 5)
        weight = 336;
      if (bucket == 6)
        weight = 204;
      if (bucket == 7)
        weight = 124;
      if (bucket == 8)
        weight = 75;
      if (bucket == 9)
        weight = 46;
      if (bucket == 10)
        weight = 28;
      if (bucket == 11)
        weight = 17;
      if (bucket == 12)
        weight = 10;
      if (bucket == 13)
        weight = 6;
      if (bucket == 14)
        weight = 4;
      if (bucket == 15)
        weight = 2;
      row_weights[j] = weight;
      weight_sum += row_weights[j];
    }

    if (weight_sum == 0)
      weight_sum = 1;

    int32_t quantized_sum = 0;
    for (int j = 0; j < SEQ_LEN; ++j) {
      int32_t q = (int32_t)((row_weights[j] * SOFTMAX_ROW_SUM + weight_sum / 2) /
                            weight_sum);
      if (q > SOFTMAX_ROW_SUM)
        q = SOFTMAX_ROW_SUM;
      probs[i][j] = (int8_t)q;
      quantized_sum += q;
    }

    int32_t residual = SOFTMAX_ROW_SUM - quantized_sum;
    int32_t corrected = (int32_t)probs[i][best_j] + residual;
    if (corrected < 0)
      corrected = 0;
    if (corrected > SOFTMAX_ROW_SUM)
      corrected = SOFTMAX_ROW_SUM;
    probs[i][best_j] = (int8_t)corrected;
  }
}

void attention32x64_int8(int32_t O[SEQ_LEN][EMBED_DIM],
                         int8_t Q[SEQ_LEN][EMBED_DIM],
                         int8_t K[SEQ_LEN][EMBED_DIM],
                         int8_t V[SEQ_LEN][EMBED_DIM]) {
  int32_t scores[SEQ_LEN][SEQ_LEN];
  int8_t probs[SEQ_LEN][SEQ_LEN];

  for (int i = 0; i < SEQ_LEN; ++i) {
    for (int j = 0; j < SEQ_LEN; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < EMBED_DIM; ++k)
        acc += (int32_t)Q[i][k] * (int32_t)K[j][k];
      scores[i][j] = acc;
    }
  }

  softmax_rows_q7(probs, scores);

  for (int i = 0; i < SEQ_LEN; ++i) {
    for (int d = 0; d < EMBED_DIM; ++d) {
      int32_t acc = 0;
      for (int j = 0; j < SEQ_LEN; ++j)
        acc += (int32_t)probs[i][j] * (int32_t)V[j][d];
      O[i][d] = acc;
    }
  }
}
