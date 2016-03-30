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
        const Real &alpha, const Real &momentum, const Real &decay,
        Layer *prev,
        Filler *weightFiller,
        Filler *biasFiller,
        Activation *activation) :

          Layer(name, prev->num, 1, 1, depth, prev),
          inWidth(prev->width), inHeight(prev->height), inDepth(prev->depth),
          inputSize(prev->width * prev->height * prev->depth),
          alpha(alpha), momentum(momentum), decay(decay),
          weightFiller(weightFiller), biasFiller(biasFiller), activation(activation) {

        weight.resize(depth * inputSize);
        weightHistory.resize(depth * inputSize);
        bias.resize(depth);
        biasHistory.resize(depth);
        delta.resize(depth * inputSize);
        biasMultiplier = Vec(num, 1.0);

        biasDelta.resize(depth);
        reshape(num, width, height, depth, &constantOutput);

        weightFiller->fill(&weight);
        biasFiller->fill(&bias);

        flatInput.resize(num * inputSize);
        flatOutput.resize(num * depth);
        flatNextErrors.resize(num * depth);
        flatErrors.resize(num * inputSize);
      }

      const int inWidth;
      const int inHeight;
      const int inDepth;
      const int inputSize;

      const Real alpha;
      const Real momentum;
      const Real decay;

      Filler *weightFiller;
      Filler *biasFiller;
      Activation *activation;

      Vec weight;
      Vec weightHistory;
      Vec delta;
      Vec bias;
      Vec biasHistory;
      Vec biasDelta;
      Vec biasMultiplier;

      vector<Vec> constantOutput;

      Vec flatInput;
      Vec flatOutput;
      Vec flatNextErrors;
      Vec flatErrors;

      void flatten(const vector<Vec> a, Vec *b) {
        int i = 0;
        for (int j = 0; j < a.size(); j++) {
          for (int k = 0; k < a[j].size(); k++) {
            b->at(i++) = a[j][k];
          }
        }
      }

      void reconstruct(const Vec &a, vector<Vec> *b) {
        int i = 0;
        for (int j = 0; j < b->size(); j++) {
          for (int k = 0; k < b->at(j).size(); k++) {
            b->at(j)[k] = a[i++];
          }
        }
      }

      void forward() {
        flatten(prev->output, &flatInput);

        gemm(
            CblasNoTrans, CblasTrans,
            num, depth, inputSize,
            1., flatInput, weight,
            0., &flatOutput);

        gemm(
            CblasNoTrans, CblasNoTrans,
            num, depth, 1,
            1., biasMultiplier, bias,
            1., &flatOutput);

        reconstruct(flatOutput, &output);
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&errors);

        flatten(nextErrors, &flatNextErrors);

        gemm(
            CblasTrans, CblasNoTrans,
            depth, inputSize, num,
            1., flatNextErrors, flatInput,
            1., &delta);

        gemv(
            CblasTrans,
            num, depth,
            1., flatNextErrors, biasMultiplier,
            1., &biasDelta);

        gemm(
            CblasNoTrans, CblasNoTrans,
            num, inputSize, depth,
            1., flatNextErrors, weight,
            0., &flatErrors);

#ifndef TESTING
        applyUpdate();
#endif

        reconstruct(flatErrors, &errors);
      }

      void applyUpdate() {
        updateParam(alpha, momentum, decay, &delta, &weight, &weightHistory);
        updateParam(alpha, momentum, decay, &biasDelta, &bias, &biasHistory);
      }
  };
}
