#pragma once

#include "layer.h"
#include "util.h"

namespace con {
  class ReluLayer : public Layer {
    public:
      ReluLayer(const string &name, Layer *prev) :
        Layer(name, prev->num, prev->width, prev->height, prev->depth, prev),
        inWidth(prev->width), inHeight(prev->height), inDepth(prev->depth) {}

      const int inWidth;
      const int inHeight;
      const int inDepth;

      void forward() {
        for (int n = 0; n < num; n++) {
          for (int out = 0; out < depth; out++) {
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                const int outputIndex = out * width * height + h * width + w;
                output[n][outputIndex] = std::max<Real>(0, prev->output[n][outputIndex]);
              }
            }
          }
        }
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          for (int out = 0; out < depth; out++) {
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                const int outputIndex = out * width * height + h * width + w;
                if (prev->output[n][outputIndex] < 0) {
                  errors[n][outputIndex] = 0;
                } else {
                  errors[n][outputIndex] = nextErrors[n][outputIndex];
                }
              }
            }
          }
        }
      }
  };
}
