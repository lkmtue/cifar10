#pragma once

#include "activation.h"
#include "filler.h"
#include "layer.h"
#include "util.h"

namespace con {
  class FullyConnectedLayer : public Layer {
    public:
      FullyConnectedLayer(
        const string &name,
        const int &depth,
        const Real &alpha, const Real &momentum,
        Layer *prev,
        Filler *weightFiller,
        Filler *biasFiller,
        Activation *activation) :

          Layer(name, prev->num, 1, 1, depth, prev),
          inWidth(prev->width), inHeight(prev->height), inDepth(prev->depth),
          inputSize(prev->width * prev->height * prev->depth),
          alpha(alpha), momentum(momentum),
          weightFiller(weightFiller), biasFiller(biasFiller), activation(activation) {

        weight.resize(depth * inputSize);
        bias.resize(depth);
        delta.resize(depth * inputSize);
        biasDelta.resize(depth);
        reshape(num, width, height, depth, &constantOutput);

        weightFiller->fill(&weight);
        biasFiller->fill(&bias);
      }

      const int inWidth;
      const int inHeight;
      const int inDepth;
      const int inputSize;

      const Real alpha;
      const Real momentum;

      Filler *weightFiller;
      Filler *biasFiller;
      Activation *activation;

      Vec weight;
      Vec bias;
      vector<Vec> constantOutput;

      void forward() {
        for (int n = 0; n < num; n++) {
          for (int outputIndex = 0; outputIndex < depth; outputIndex++) {
            Real result = 0;
            for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
              const int weightIndex = outputIndex * inputSize + inputIndex;

              result += weight[weightIndex] * prev->output[n][inputIndex];
            }

            constantOutput[n][outputIndex] = result + bias[outputIndex];
            output[n][outputIndex] = activation->f(constantOutput[n][outputIndex]);
          }
        }
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        for (int n = 0; n < num; n++) {
          for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            for (int outputIndex = 0; outputIndex < depth; outputIndex++) {

              const int weightIndex = outputIndex * inputSize + inputIndex;

              errors[n][inputIndex] +=
                nextErrors[n][outputIndex] *
                  // activation->df(output[n][outputIndex]) *
                  activation->df(constantOutput[n][outputIndex]) *
                  weight[weightIndex];
            }
          }
        }

        for (int outputIndex = 0; outputIndex < depth; outputIndex++) {
          for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
            const int weightIndex = outputIndex * inputSize + inputIndex;

            Real d = momentum * delta[weightIndex];

            for (int n = 0; n < num; n++) {
              const Real dedw =
                nextErrors[n][outputIndex] *
                  // activation->df(output[n][outputIndex]) *
                  activation->df(constantOutput[n][outputIndex]) *
                  prev->output[n][inputIndex];

              d += alpha * dedw;
            }

#ifndef TESTING
            weight[weightIndex] -= d;
#endif
            delta[weightIndex] = d;
          }

          biasDelta[outputIndex] = 0;
          for (int n = 0; n < num; n++) {
            // biasDelta[outputIndex] += alpha * nextErrors[n][outputIndex] * activation->df(output[n][outputIndex]);
            biasDelta[outputIndex] += alpha * nextErrors[n][outputIndex] * activation->df(constantOutput[n][outputIndex]);
#ifndef TESTING
            // bias[outputIndex] -= alpha * nextErrors[n][outputIndex] * activation->df(output[n][outputIndex]);
            bias[outputIndex] -= alpha * nextErrors[n][outputIndex] * activation->df(constantOutput[n][outputIndex]);
#endif
          }
        }
      }

      Vec delta;
      Vec biasDelta;
  };
}
