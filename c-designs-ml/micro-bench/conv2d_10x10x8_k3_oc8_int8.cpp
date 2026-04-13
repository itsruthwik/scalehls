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

// Equivalent im2col GEMM:
//   M = 8
//   K = 72
//   N = 64
extern "C" void conv2d_10x10x8_k3_oc8_int8(int8_t O[N][OC][OH][OW],
                                           int8_t I[N][IC][IH][IW],
                                           int8_t W[OC][IC][KH][KW]) {
  for (int n = 0; n < N; ++n) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          // int32_t acc = (int32_t)O[n][oc][oh][ow];
          int32_t acc = 0;
          for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                acc += (int32_t)I[n][ic][oh + kh][ow + kw] *
                       (int32_t)W[oc][ic][kh][kw];
              }
            }
          }
          O[n][oc][oh][ow] = (int8_t)acc;
        }
      }
    }
  }
}
