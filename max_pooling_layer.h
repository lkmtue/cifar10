#pragma once

#include "layer.h"
#include "util.h"

namespace con {
  class MaxPoolingLayer : public Layer {
    public:
      MaxPoolingLayer(const string &name, const int &kernel, const int &stride, Layer *prev) :
        Layer(
          name,
          prev->num,
          ceilDiv(prev->width - kernel, stride) + 1,
          ceilDiv(prev->height - kernel, stride) + 1,
          prev->depth,
          prev),
        kernel(kernel), stride(stride) {

        reshape(num, width, height, depth, &maxIndex);
      }

      const int kernel;
      const int stride;

      void forward() {
        for (int n = 0; n < num; n++) {
          for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                int index = getIndex(d, h, w);
                output[n][index] = getMax(n, d, h, w, index);
              }
            }
          }
        }
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          for (int i = 0; i < depth * height * width; i++) {
            errors[n][maxIndex[n][i]] += nextErrors[n][i];
          }
        }
      }

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {}

    private:
      vector<Vec> maxIndex;

      Real getMax(const int &n, const int &d, const int &h, const int &w, const int &outIndex) {
        int startH = h * stride;
        int startW = w * stride;

        int pos = -1;

        for (int i = startH; i < startH + kernel && i < inHeight; i++) {
          for (int j = startW; j < startW + kernel && j < inWidth; j++) {
            int index = d * inHeight * inWidth + i * inWidth + j;
            if (pos == -1 || prev->output[n][index] > prev->output[n][pos]) {
              pos = index;
            }
          }
        }

        maxIndex[n][outIndex] = pos;

        return prev->output[n][pos];
      }
  };
}
