#pragma once

#include "layer.h"
#include "util.h"

namespace con {
  class AveragePoolingLayer : public Layer {
    public:
      AveragePoolingLayer(const string &name, const int &kernel, const int &stride, Layer *prev) :
        Layer(
          name,
          prev->num,
          ceilDiv(prev->width - kernel, stride) + 1,
          ceilDiv(prev->height - kernel, stride) + 1,
          prev->depth,
          prev),
        kernel(kernel), stride(stride) {}

      const int kernel;
      const int stride;

      void forward() {
        for (int n = 0; n < num; n++) {
          for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                int index = getIndex(d, h, w);
                output[n][index] = getSum(n, d, h, w, index);
              }
            }
          }
        }
      }

      Real getSum(const int &n, const int &d, const int &h, const int &w, const int &outIndex) {
        const int startH = h * stride;
        const int startW = w * stride;

        const int endH = std::min(startH + kernel, inHeight);
        const int endW = std::min(startW + kernel, inWidth);

        const int area = (endH - startH) * (endW - startW);

        Real sum = 0 ;

        for (int i = startH; i < endH; i++) {
          for (int j = startW; j < endW; j++) {
            const int inputIndex = d * inHeight * inWidth + i * inWidth + j;
            sum += prev->output[n][inputIndex];
          }
        }

        return sum / area;
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          for (int out = 0; out < depth; out++) {
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                const int outputIndex = out * width * height + h * width + w;

                const int startH = h * stride;
                const int startW = w * stride;

                const int endH = std::min(startH + kernel, inHeight);
                const int endW = std::min(startW + kernel, inWidth);

                const int area = (endH - startH) * (endW - startW);

                for (int i = startH; i < endH; i++) {
                  for (int j = startW; j < endW; j++) {
                    const int inputIndex = out * inHeight * inWidth + i * inWidth + j;

                    errors[n][inputIndex] += nextErrors[n][outputIndex] / area;
                  }
                }
              }
            }
          }
        }
      }

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {}
  };
}
