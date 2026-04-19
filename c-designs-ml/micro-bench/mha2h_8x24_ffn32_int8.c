/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define SEQ_LEN 8
#define MODEL_DIM 24
#define NUM_HEADS 2
#define HEAD_DIM 12
#define FFN_HIDDEN_DIM 32
#define SOFTMAX_Q_BITS 7
#define SOFTMAX_ROW_SUM ((1 << SOFTMAX_Q_BITS) - 1)
#define SOFTMAX_SCORE_SCALE 4
#define SOFTMAX_BUCKET_STRIDE 8
#define SOFTMAX_LUT_LAST 16

// Equivalent GEMMs:
//   Head 0 Q projection : M = 12, K = 24, N = 8
//   Head 0 K projection : M = 12, K = 24, N = 8
//   Head 0 V projection : M = 12, K = 24, N = 8
//   Head 1 Q projection : M = 12, K = 24, N = 8
//   Head 1 K projection : M = 12, K = 24, N = 8
//   Head 1 V projection : M = 12, K = 24, N = 8
//   Head 0 scores       : M = 8,  K = 12, N = 8
//   Head 0 values       : M = 8,  K = 8,  N = 12
//   Head 1 scores       : M = 8,  K = 12, N = 8
//   Head 1 values       : M = 8,  K = 8,  N = 12
//   Output projection   : M = 24, K = 24, N = 8
//   FFN layer 1         : M = 32, K = 24, N = 8
//   FFN layer 2         : M = 24, K = 32, N = 8

static void softmax_rows_q7_8(int8_t probs[SEQ_LEN][SEQ_LEN],
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

void mha2h_8x24_ffn32_int8(int8_t O[MODEL_DIM][SEQ_LEN],
                           int8_t WQ0[HEAD_DIM][MODEL_DIM],
                           int8_t WK0[HEAD_DIM][MODEL_DIM],
                           int8_t WV0[HEAD_DIM][MODEL_DIM],
                           int8_t WQ1[HEAD_DIM][MODEL_DIM],
                           int8_t WK1[HEAD_DIM][MODEL_DIM],
                           int8_t WV1[HEAD_DIM][MODEL_DIM],
                           int8_t WO[MODEL_DIM][MODEL_DIM],
                           int8_t W1[FFN_HIDDEN_DIM][MODEL_DIM],
                           int8_t W2[MODEL_DIM][FFN_HIDDEN_DIM],
                           int8_t X[MODEL_DIM][SEQ_LEN]) {
  int8_t CONCAT[MODEL_DIM][SEQ_LEN];
  int32_t MHA_PROJ[MODEL_DIM][SEQ_LEN];
  int8_t MHA[MODEL_DIM][SEQ_LEN];
  int32_t FFN1_PROJ[FFN_HIDDEN_DIM][SEQ_LEN];
  int8_t FFN1[FFN_HIDDEN_DIM][SEQ_LEN];
  int32_t FFN2_PROJ[MODEL_DIM][SEQ_LEN];

  int8_t Q0[HEAD_DIM][SEQ_LEN];
  int8_t K0[HEAD_DIM][SEQ_LEN];
  int8_t V0[HEAD_DIM][SEQ_LEN];
  int32_t Q0_PROJ[HEAD_DIM][SEQ_LEN];
  int32_t K0_PROJ[HEAD_DIM][SEQ_LEN];
  int32_t V0_PROJ[HEAD_DIM][SEQ_LEN];
  int8_t Q0_T[SEQ_LEN][HEAD_DIM];
  int8_t V0_T[SEQ_LEN][HEAD_DIM];
  int8_t Q1[HEAD_DIM][SEQ_LEN];
  int8_t K1[HEAD_DIM][SEQ_LEN];
  int8_t V1[HEAD_DIM][SEQ_LEN];
  int32_t Q1_PROJ[HEAD_DIM][SEQ_LEN];
  int32_t K1_PROJ[HEAD_DIM][SEQ_LEN];
  int32_t V1_PROJ[HEAD_DIM][SEQ_LEN];
  int8_t Q1_T[SEQ_LEN][HEAD_DIM];
  int8_t V1_T[SEQ_LEN][HEAD_DIM];
  int32_t SCORES0[SEQ_LEN][SEQ_LEN];
  int8_t PROBS0[SEQ_LEN][SEQ_LEN];
  int32_t HEAD0_GEMM[SEQ_LEN][HEAD_DIM];
  int32_t SCORES1[SEQ_LEN][SEQ_LEN];
  int8_t PROBS1[SEQ_LEN][SEQ_LEN];
  int32_t HEAD1_GEMM[SEQ_LEN][HEAD_DIM];

  // Head 0 Q projection GEMM: Q0 = WQ0 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WQ0[m][k] * (int32_t)X[k][n];
      Q0_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = Q0_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      Q0[m][n] = (int8_t)value;
    }
  }

  // Head 0 K projection GEMM: K0 = WK0 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WK0[m][k] * (int32_t)X[k][n];
      K0_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = K0_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      K0[m][n] = (int8_t)value;
    }
  }

  // Head 0 V projection GEMM: V0 = WV0 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WV0[m][k] * (int32_t)X[k][n];
      V0_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = V0_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      V0[m][n] = (int8_t)value;
    }
  }

  // Head 1 Q projection GEMM: Q1 = WQ1 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WQ1[m][k] * (int32_t)X[k][n];
      Q1_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = Q1_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      Q1[m][n] = (int8_t)value;
    }
  }

  // Head 1 K projection GEMM: K1 = WK1 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WK1[m][k] * (int32_t)X[k][n];
      K1_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = K1_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      K1[m][n] = (int8_t)value;
    }
  }

  // Head 1 V projection GEMM: V1 = WV1 * X.
  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WV1[m][k] * (int32_t)X[k][n];
      V1_PROJ[m][n] = acc;
    }
  }

  for (int m = 0; m < HEAD_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = V1_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      V1[m][n] = (int8_t)value;
    }
  }

  // Head 0 packing: transpose Q and V slices into seq-major form.
  for (int i = 0; i < SEQ_LEN; ++i) {
    for (int d = 0; d < HEAD_DIM; ++d) {
      Q0_T[i][d] = Q0[d][i];
      V0_T[i][d] = V0[d][i];
    }
  }

  // Head 0 score GEMM: SCORES0 = Q0_T * K0.
  for (int m = 0; m < SEQ_LEN; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < HEAD_DIM; ++k)
        acc += (int32_t)Q0_T[m][k] * (int32_t)K0[k][n];
      SCORES0[m][n] = acc;
    }
  }

  // Head 0 softmax over the score rows.
  softmax_rows_q7_8(PROBS0, SCORES0);

  // Head 0 value GEMM: HEAD0_GEMM = PROBS0 * V0_T.
  for (int m = 0; m < SEQ_LEN; ++m) {
    for (int n = 0; n < HEAD_DIM; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < SEQ_LEN; ++k)
        acc += (int32_t)PROBS0[m][k] * (int32_t)V0_T[k][n];
      HEAD0_GEMM[m][n] = acc;
    }
  }

  // Head 1 packing: transpose Q and V slices into seq-major form.
  for (int i = 0; i < SEQ_LEN; ++i) {
    for (int d = 0; d < HEAD_DIM; ++d) {
      Q1_T[i][d] = Q1[d][i];
      V1_T[i][d] = V1[d][i];
    }
  }

  // Head 1 score GEMM: SCORES1 = Q1_T * K1.
  for (int m = 0; m < SEQ_LEN; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < HEAD_DIM; ++k)
        acc += (int32_t)Q1_T[m][k] * (int32_t)K1[k][n];
      SCORES1[m][n] = acc;
    }
  }

  // Head 1 softmax over the score rows.
  softmax_rows_q7_8(PROBS1, SCORES1);

  // Head 1 value GEMM: HEAD1_GEMM = PROBS1 * V1_T.
  for (int m = 0; m < SEQ_LEN; ++m) {
    for (int n = 0; n < HEAD_DIM; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < SEQ_LEN; ++k)
        acc += (int32_t)PROBS1[m][k] * (int32_t)V1_T[k][n];
      HEAD1_GEMM[m][n] = acc;
    }
  }

  // Concatenate heads into a MODEL_DIM x SEQ_LEN activation tensor.
  for (int i = 0; i < SEQ_LEN; ++i) {
    for (int d = 0; d < HEAD_DIM; ++d) {
      int32_t head0_value = HEAD0_GEMM[i][d];
      int32_t head1_value = HEAD1_GEMM[i][d];
      head0_value = (head0_value + 64) >> 7;
      if (head0_value > 127)
        head0_value = 127;
      if (head0_value < -128)
        head0_value = -128;
      head1_value = (head1_value + 64) >> 7;
      if (head1_value > 127)
        head1_value = 127;
      if (head1_value < -128)
        head1_value = -128;
      CONCAT[d][i] = (int8_t)head0_value;
      CONCAT[HEAD_DIM + d][i] = (int8_t)head1_value;
    }
  }

  // Output projection GEMM: MHA_PROJ = WO * CONCAT.
  for (int m = 0; m < MODEL_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)WO[m][k] * (int32_t)CONCAT[k][n];
      MHA_PROJ[m][n] = acc;
    }
  }

  // Output projection epilogue: add a residual connection from X.
  for (int m = 0; m < MODEL_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = MHA_PROJ[m][n] + (int32_t)X[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      MHA[m][n] = (int8_t)value;
    }
  }

  // FFN layer 1 GEMM: FFN1_PROJ = W1 * MHA.
  for (int m = 0; m < FFN_HIDDEN_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < MODEL_DIM; ++k)
        acc += (int32_t)W1[m][k] * (int32_t)MHA[k][n];
      FFN1_PROJ[m][n] = acc;
    }
  }

  // FFN layer 1 epilogue: apply ReLU.
  for (int m = 0; m < FFN_HIDDEN_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = FFN1_PROJ[m][n];
      value = (value + 64) >> 7;
      if (value < 0)
        value = 0;
      if (value > 127)
        value = 127;
      FFN1[m][n] = (int8_t)value;
    }
  }

  // FFN layer 2 GEMM: FFN2_PROJ = W2 * FFN1.
  for (int m = 0; m < MODEL_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < FFN_HIDDEN_DIM; ++k)
        acc += (int32_t)W2[m][k] * (int32_t)FFN1[k][n];
      FFN2_PROJ[m][n] = acc;
    }
  }

  // FFN layer 2 epilogue: add residual from MHA.
  for (int m = 0; m < MODEL_DIM; ++m) {
    for (int n = 0; n < SEQ_LEN; ++n) {
      int32_t value = FFN2_PROJ[m][n] + (int32_t)MHA[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      O[m][n] = (int8_t)value;
    }
  }
}
