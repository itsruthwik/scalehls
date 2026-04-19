/*
 * Author: Ruthwik Reddy Sunketa
 */

#include <stdint.h>

#define N 1
#define IC 8
#define IH 10
#define IW 10
#define OC 8
#define KH 3
#define KW 3
#define OH 8
#define OW 8

void conv2d_10x10x8_k3_oc8_int8(int8_t O[N][OC][OH][OW],
                                int8_t I[N][IC][IH][IW],
                                int8_t W[OC][IC][KH][KW]) {
  int32_t ACC[N][OC][OH][OW];

  for (int n = 0; n < N; ++n) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          int32_t acc = 0;
          for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                acc += (int32_t)I[n][ic][oh + kh][ow + kw] *
                       (int32_t)W[oc][ic][kh][kw];
              }
            }
          }
          ACC[n][oc][oh][ow] = acc;
        }
      }
    }
  }

  for (int n = 0; n < N; ++n) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          int32_t value = ACC[n][oc][oh][ow];
          value = (value + 64) >> 7;
          if (value > 127)
            value = 127;
          if (value < -128)
            value = -128;
          O[n][oc][oh][ow] = (int8_t)value;
        }
      }
    }
  }
}
