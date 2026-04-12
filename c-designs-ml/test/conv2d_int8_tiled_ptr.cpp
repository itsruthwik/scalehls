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

#define TOC 1
#define TOH 1
#define TOW 1

extern "C" void conv2d_int8_tiled_ptr(int8_t *O, int8_t *I, int8_t *W) {
  for (int oco = 0; oco < OC; oco += TOC) {
    for (int oho = 0; oho < OH; oho += TOH) {
      for (int owo = 0; owo < OW; owo += TOW) {
        for (int oc = oco; oc < oco + TOC; ++oc) {
          for (int oh = oho; oh < oho + TOH; ++oh) {
            for (int ow = owo; ow < owo + TOW; ++ow) {
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
    }
  }
}
