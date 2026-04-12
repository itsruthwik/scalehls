#include <math.h>

#define BATCH 1
#define TOKENS 8
#define DEPTH 8
#define VALUE_DEPTH 8

static void softmax_rows_fp32(float probs[TOKENS][TOKENS],
                              float scores[TOKENS][TOKENS]) {
  for (int i = 0; i < TOKENS; ++i) {
    float row_max = scores[i][0];
    for (int j = 1; j < TOKENS; ++j) {
      if (scores[i][j] > row_max)
        row_max = scores[i][j];
    }

    float row_sum = 0.0f;
    for (int j = 0; j < TOKENS; ++j) {
      float shifted = scores[i][j] - row_max;
      float value = expf(shifted);
      probs[i][j] = value;
      row_sum += value;
    }

    for (int j = 0; j < TOKENS; ++j)
      probs[i][j] /= row_sum;
  }
}

extern "C" void attention_fp32(float O[BATCH][TOKENS][VALUE_DEPTH],
                                float Q[BATCH][TOKENS][DEPTH],
                                float K[BATCH][TOKENS][DEPTH],
                                float V[BATCH][TOKENS][VALUE_DEPTH]) {
  float scores[TOKENS][TOKENS] = {0.0f};
  float probs[TOKENS][TOKENS] = {0.0f};

  for (int b = 0; b < BATCH; ++b) {
    for (int i = 0; i < TOKENS; ++i) {
      for (int j = 0; j < TOKENS; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < DEPTH; ++k)
          sum += Q[b][i][k] * K[b][j][k];
        scores[i][j] = sum;
      }
    }

    softmax_rows_fp32(probs, scores);

    for (int i = 0; i < TOKENS; ++i) {
      for (int d = 0; d < VALUE_DEPTH; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < TOKENS; ++j)
          sum += probs[i][j] * V[b][j][d];
        O[b][i][d] = sum;
      }
    }
  }
}
