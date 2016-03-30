#pragma once

namespace con {
  class Param {
    public:
      Param(const int &size, Filler *filler) {
        value.resize(size);
        filler->fill(&value);

        delta.resize(size);
        history.resize(size);
      }

      Vec value;
      Vec delta;
      Vec history;

      void update(const Real &lr, const Real &momentum, const Real &decay) {
        // Decay.
        for (int i = 0; i < delta.size(); i++) {
          delta[i] += decay * value[i];
        }

        // Adagrad.
        for (int i = 0; i < delta.size(); i++) {
          history[i] += sqr(delta[i]);
          value[i] += -lr * delta[i] / (sqrt(history[i]) + 1e-7);
        }
      }
  };
}
