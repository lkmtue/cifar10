#pragma once

#include "layer.h"

namespace con {
  class InputLayer : public Layer {
    public:
      InputLayer(const string &name, const int &num, const int &width, const int &height, const int &depth) :
        Layer(name, num, width, height, depth, nullptr) {
      }

      void setOutput(const vector<Vec> &o) {
        output = o;
      }

      void forward() {}

      void backProp(const vector<Vec> &nextErrors) {}

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {}
  };
}
