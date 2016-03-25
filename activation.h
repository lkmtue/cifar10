#pragma once

#include "util.h"

namespace con {
  class Activation {
    public:
      Activation() {}
      virtual Real f(const Real &x) = 0;
      virtual Real df(const Real &x) = 0;
  };

  class ConstantActivation : public Activation {
    public:
      ConstantActivation(): Activation() {}

      Real f(const Real &x) {
        return x;
      }

      Real df(const Real &x) {
        return 1;
      }
  };

  class SigmoidActivation : public Activation {
    public:
      SigmoidActivation(): Activation() {}

      Real f(const Real &x) {
        return sigmoid(x);
      }

      Real df(const Real &x) {
        Real s = sigmoid(x);
        return s * (1.0 - s);
      }
  };
}
