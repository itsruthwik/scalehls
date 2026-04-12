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

extern "C" void conv2d_int8_ptr(int8_t *O, int8_t *I, int8_t *W) {
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              int out_idx = ((oc * OH) + oh) * OW + ow;
              int in_idx = (((ic * IH) + (oh + kh)) * IW) + (ow + kw);
              int w_idx = (((oc * IC) + ic) * KH + kh) * KW + kw;
              int32_t in = (int32_t)I[in_idx];
              int32_t w = (int32_t)W[w_idx];
              int32_t prod = in * w;
              int32_t out = (int32_t)O[out_idx];
              int32_t sum = out + prod;
              O[out_idx] = (int8_t)sum;
            }
          }
        }
      }
    }
  }
}
