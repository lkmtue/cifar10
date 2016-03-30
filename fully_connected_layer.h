#pragma once

#include "filler.h"
#include "layer.h"
#include "util.h"

namespace con {
  class FullyConnectedLayer : public Layer {
    public:
      FullyConnectedLayer(
        const string &name,
        const int &depth,
        Layer *prev,
        Filler *weightFiller,
        Filler *biasFiller) :

          Layer(name, prev->num, 1, 1, depth, prev),
          inputSize(prev->width * prev->height * prev->depth),
          weight(depth * inputSize, weightFiller),
          bias(depth, biasFiller) {

        biasMultiplier = Vec(num, 1.0);

        flatInput.resize(num * inputSize);
        flatOutput.resize(num * depth);
        flatNextErrors.resize(num * depth);
        flatErrors.resize(num * inputSize);
      }

      const int inputSize;

      Param weight;
      Param bias;

      Vec biasMultiplier;

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
            1., flatInput, weight.value,
            0., &flatOutput);

        gemm(
            CblasNoTrans, CblasNoTrans,
            num, depth, 1,
            1., biasMultiplier, bias.value,
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
            1., &weight.delta);

        gemv(
            CblasTrans,
            num, depth,
            1., flatNextErrors, biasMultiplier,
            1., &bias.delta);

        gemm(
            CblasNoTrans, CblasNoTrans,
            num, inputSize, depth,
            1., flatNextErrors, weight.value,
            0., &flatErrors);

        reconstruct(flatErrors, &errors);
      }

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {
        weight.update(lr, momentum, decay);
        bias.update(lr, momentum, decay);
      }
  };
}
