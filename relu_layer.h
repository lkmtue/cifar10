#pragma once

#include "layer.h"
#include "util.h"

namespace con {
  class ReluLayer : public Layer {
    public:
      ReluLayer(const string &name, Layer *prev) :
        Layer(name, prev->num, prev->width, prev->height, prev->depth, prev),
        inputSize(prev->depth * prev->width * prev->height) {}

      const int inputSize;

      void forward() {
        for (int n = 0; n < num; n++) {
          for (int i = 0; i < inputSize; i++) {
            output[n][i] = std::max<Real>(0, prev->output[n][i]);
          }
        }
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          for (int i = 0; i < inputSize; i++) {
            if (prev->output[n][i] < 0) {
              errors[n][i] = 0;
            } else {
              errors[n][i] = nextErrors[n][i];
            }
          }
        }
      }

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {}
  };
}
