#pragma once

#include "layer.h"
#include "util.h"

namespace con {
  class SoftmaxLossLayer : public Layer {
    public:
      SoftmaxLossLayer(const string &name, Layer *prev) : Layer(name, prev->num, 1, 1, prev->depth, prev) {
        reshape(num, 1, 1, depth, &e);
        reshape(num, 1, 1, depth, &subtract);
        sumE.resize(num);
        maxProb.resize(num);
      }

      void getResults(vector<int> *results) {
        results->clear();
        for (int i = 0; i < num; i++) {
          results->push_back(getResult(i));
        }
      }

      int getResult(const int &i) {
        int result = 0;
        for (int j = 0; j < depth; j++) {
          if (output[i][j] > output[i][result]) {
            result = j;
          }
        }
        return result;
      }

      void setLabels(const vector<int> &l) {
        labels = l;
      }

      Real loss() {
        Real loss = 0;

        for (int n = 0; n < num; n++) {
          maxProb[n] = 0;
          for (int i = 1; i < depth; i++) {
            if (prev->output[n][i] > prev->output[n][maxProb[n]]) {
              maxProb[n] = i;
            }
          }

          sumE[n] = 0;
          for (int i = 0; i < depth; i++) {
            subtract[n][i] = prev->output[n][i] - prev->output[n][maxProb[n]];

            e[n][i] = exp(subtract[n][i]);

            sumE[n] += e[n][i];
          }

          for (int i = 0; i < depth; i++) {
            output[n][i] = e[n][i] / sumE[n];
          }

          loss -= log(output[n][labels[n]]);
        }

        loss /= num;

        return loss;
      }

      void forward() {
        l = loss();
      }

      // d(log(x)) / d(x)
      Real logDerivative(const Real &x) {
        return 1.0 / x;
      }

      // d(x / (x + a)) / d(x)
      Real fractionXADerivative(const Real &x, const Real &a) {
        return a / sqr(x + a);
      }

      // d(a / (x + b)) / d(x)
      Real fractionXABDerivative(const Real &x, const Real &a, const Real &b) {
        return -a / sqr(x + b);
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          const int label = labels[n];
          const Real dlog = -logDerivative(e[n][label] / sumE[n]) / num;

          for (int i = 0; i < depth; i++) {
            const Real dexplabel =
              i == label ?
                fractionXADerivative(e[n][label], sumE[n] - e[n][label]) :
                fractionXABDerivative(e[n][i], e[n][label], sumE[n] - e[n][i]);

            const Real dexpsubtract = exp(subtract[n][i]);

            const Real dsubtract = i == maxProb[n] ? 1 : 1;

            errors[n][i] = dlog * dexplabel * dexpsubtract * dsubtract;
          }
        }
      }

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {}

      vector<int> labels;

      vector<int> maxProb;
      vector<Vec> e;
      vector<Vec> subtract;
      Vec sumE;
      Real l;
  };
}
