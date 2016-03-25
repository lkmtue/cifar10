#pragma once

#include <algorithm>

#include "util.h"

namespace con {
  class Filler {
    public:
      Filler() {}
      virtual void fill(Vec *a) = 0;
  };

  class GaussianFiller : public Filler {
    public:
      GaussianFiller(const Real &mean, const Real &std): Filler(), mean(mean), std(std) {}

      void fill(Vec *a) {
        gaussianRng(mean, std, a);
      }

      const Real mean;
      const Real std;
  };

  class UniformFiller : public Filler {
    public:
      UniformFiller(const Real &min, const Real &max): Filler(), min(min), max(max) {}

      void fill(Vec *a) {
        randomizeVec(min, max, a);
      }

      const Real min;
      const Real max;
  };

  class ConstantFiller: public Filler {
    public:
      ConstantFiller(const Real &c): Filler(), c(c) {}

      void fill(Vec *a) {
        std::fill(a->begin(), a->end(), c);
      }

      const Real c;
    };
}
