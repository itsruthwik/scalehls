/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define IC0 8
#define IH0 6
#define IW0 6

#define OC1 8
#define KH1 1
#define KW1 1
#define OH1 6
#define OW1 6

#define OC2 8
#define KH2 3
#define KW2 3
#define OH2 4
#define OW2 4

#define PH 2
#define PW 2
#define SH 2
#define SW 2
#define OH_POOL 2
#define OW_POOL 2

#define OC3 16
#define KH3 1
#define KW3 1
#define OH3 2
#define OW3 2

#define FLAT 64
#define DENSE_OUT 16
#define DENSE_BATCH 1

/*
 * CNN micro-bench:
 *   1. input:      6x6x8
 *   2. 1x1 conv:   6x6x8  -> 6x6x8
 *   3. ReLU
 *   4. 3x3 conv:   6x6x8  -> 4x4x8 (valid)
 *   5. ReLU
 *   6. 2x2 maxpool 4x4x8  -> 2x2x8
 *   7. 1x1 conv:   2x2x8  -> 2x2x16
 *   8. ReLU
 *   9. flatten:    2x2x16 -> 64x1
 *  10. dense:      64x1   -> 16x1
 *
 * The convolution stages are written as natural direct-convolution loop nests,
 * with ReLU, maxpool, flatten, and dense logic kept in separate loops.
 *
 * Equivalent GEMMs after lowering:
 *   Conv 1: M = 8,  K = 8,  N = 36
 *   Conv 2: M = 8,  K = 72, N = 16
 *   Conv 3: M = 16, K = 8,  N = 4
 *   Dense : M = 16, K = 64, N = 1
 */
void cnn5_6x6x8_oc16_int8(int8_t O[DENSE_OUT][DENSE_BATCH],
                          int8_t I[IC0][IH0][IW0],
                          int8_t W1[OC1][IC0][KH1][KW1],
                          int8_t W2[OC2][OC1][KH2][KW2],
                          int8_t W3[OC3][OC2][KH3][KW3],
                          int8_t W4[DENSE_OUT][FLAT]) {
  int32_t Y1[OC1][OH1][OW1];
  int8_t A1[OC1][OH1][OW1];
  int32_t Y2[OC2][OH2][OW2];
  int8_t A2[OC2][OH2][OW2];
  int8_t P2[OC2][OH_POOL][OW_POOL];
  int32_t Y3[OC3][OH3][OW3];
  int8_t A3[OC3][OH3][OW3];
  int8_t X4[FLAT][DENSE_BATCH];
  int32_t O_ACC[DENSE_OUT][DENSE_BATCH];

  // Stage 1: direct 1x1 convolution.
  for (int oc = 0; oc < OC1; ++oc) {
    for (int oh = 0; oh < OH1; ++oh) {
      for (int ow = 0; ow < OW1; ++ow) {
        int32_t acc = 0;
        for (int ic = 0; ic < IC0; ++ic)
          acc += (int32_t)W1[oc][ic][0][0] * (int32_t)I[ic][oh][ow];
        Y1[oc][oh][ow] = acc;
      }
    }
  }

  // Stage 2: ReLU.
  for (int oc = 0; oc < OC1; ++oc) {
    for (int oh = 0; oh < OH1; ++oh) {
      for (int ow = 0; ow < OW1; ++ow) {
        int32_t value = Y1[oc][oh][ow];
        value = (value + 64) >> 7;
        if (value < 0)
          value = 0;
        if (value > 127)
          value = 127;
        A1[oc][oh][ow] = (int8_t)value;
      }
    }
  }

  // Stage 3: direct 3x3 valid convolution.
  for (int oc = 0; oc < OC2; ++oc) {
    for (int oh = 0; oh < OH2; ++oh) {
      for (int ow = 0; ow < OW2; ++ow) {
        int32_t acc = 0;
        for (int ic = 0; ic < OC1; ++ic) {
          for (int kh = 0; kh < KH2; ++kh) {
            for (int kw = 0; kw < KW2; ++kw) {
              acc += (int32_t)W2[oc][ic][kh][kw] *
                     (int32_t)A1[ic][oh + kh][ow + kw];
            }
          }
        }
        Y2[oc][oh][ow] = acc;
      }
    }
  }

  // Stage 4: ReLU.
  for (int oc = 0; oc < OC2; ++oc) {
    for (int oh = 0; oh < OH2; ++oh) {
      for (int ow = 0; ow < OW2; ++ow) {
        int32_t value = Y2[oc][oh][ow];
        value = (value + 64) >> 7;
        if (value < 0)
          value = 0;
        if (value > 127)
          value = 127;
        A2[oc][oh][ow] = (int8_t)value;
      }
    }
  }

  // Stage 5: 2x2 maxpool with stride 2.
  for (int oc = 0; oc < OC2; ++oc) {
    for (int oh = 0; oh < OH_POOL; ++oh) {
      for (int ow = 0; ow < OW_POOL; ++ow) {
        int32_t max_value = (int32_t)A2[oc][oh * SH][ow * SW];
        for (int kh = 0; kh < PH; ++kh) {
          for (int kw = 0; kw < PW; ++kw) {
            int32_t value = (int32_t)A2[oc][oh * SH + kh][ow * SW + kw];
            if (value > max_value)
              max_value = value;
          }
        }
        P2[oc][oh][ow] = (int8_t)max_value;
      }
    }
  }

  // Stage 6: direct 1x1 convolution on pooled activations.
  for (int oc = 0; oc < OC3; ++oc) {
    for (int oh = 0; oh < OH3; ++oh) {
      for (int ow = 0; ow < OW3; ++ow) {
        int32_t acc = 0;
        for (int ic = 0; ic < OC2; ++ic)
          acc += (int32_t)W3[oc][ic][0][0] * (int32_t)P2[ic][oh][ow];
        Y3[oc][oh][ow] = acc;
      }
    }
  }

  // Stage 7: ReLU.
  for (int oc = 0; oc < OC3; ++oc) {
    for (int oh = 0; oh < OH3; ++oh) {
      for (int ow = 0; ow < OW3; ++ow) {
        int32_t value = Y3[oc][oh][ow];
        value = (value + 64) >> 7;
        if (value < 0)
          value = 0;
        if (value > 127)
          value = 127;
        A3[oc][oh][ow] = (int8_t)value;
      }
    }
  }

  // Stage 8: flatten 2x2x16 into 64x1 in channel-major order.
  for (int oc = 0; oc < OC3; ++oc) {
    for (int oh = 0; oh < OH3; ++oh) {
      for (int ow = 0; ow < OW3; ++ow) {
        int k = (oc * OH3 + oh) * OW3 + ow;
        X4[k][0] = A3[oc][oh][ow];
      }
    }
  }

  // Stage 9: dense layer.
  for (int m = 0; m < DENSE_OUT; ++m) {
    for (int n = 0; n < DENSE_BATCH; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < FLAT; ++k)
        acc += (int32_t)W4[m][k] * (int32_t)X4[k][n];
      O_ACC[m][n] = acc;
    }
  }

  // Stage 10: dense epilogue with inline saturation.
  for (int m = 0; m < DENSE_OUT; ++m) {
    for (int n = 0; n < DENSE_BATCH; ++n) {
      int32_t value = O_ACC[m][n];
      value = (value + 64) >> 7;
      if (value > 127)
        value = 127;
      if (value < -128)
        value = -128;
      O[m][n] = (int8_t)value;
    }
  }
}
