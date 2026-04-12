#include <stdint.h>

#define N 1
#define IC 2
#define OC 3
#define IH 5
#define IW 5
#define KH 3
#define KW 3
#define OH (IH - KH + 1)
#define OW (IW - KW + 1)

extern "C" void conv2d_int8(int8_t O[N][OC][OH][OW],
                             int8_t I[N][IC][IH][IW],
                             int8_t W[OC][IC][KH][KW]) {
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              int32_t in = (int32_t)I[0][ic][oh + kh][ow + kw];
              int32_t w = (int32_t)W[oc][ic][kh][kw];
              int32_t prod = in * w;
              int32_t out = (int32_t)O[0][oc][oh][ow];
              int32_t sum = out + prod;
              O[0][oc][oh][ow] = (int8_t)sum;
            }
          }
        }
      }
    }
  }
}
