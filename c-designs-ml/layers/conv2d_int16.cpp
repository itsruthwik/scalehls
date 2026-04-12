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

extern "C" void conv2d_int16(int16_t O[N][OC][OH][OW],
                              int16_t I[N][IC][IH][IW],
                              int16_t W[OC][IC][KH][KW]) {
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              O[0][oc][oh][ow] += I[0][ic][oh + kh][ow + kw] * W[oc][ic][kh][kw];
            }
          }
        }
      }
    }
  }
}
