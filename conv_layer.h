#pragma once

#include "filler.h"
#include "im2col.h"
#include "layer.h"
#include "util.h"

namespace con {
  class ConvolutionalLayer : public Layer {
    public:
      ConvolutionalLayer(
        const string &name,
        const int &depth, const int &kernel, const int &stride, const int &padding,
        const Real &alpha, const Real &momentum, const Real &decay,
        Layer *prev,
        Filler *weightFiller,
        Filler *biasFiller) :

          Layer(
            name,
            prev->num,
            (prev->width - kernel + 2 * padding) / stride + 1,
            (prev->height - kernel + 2 * padding) / stride + 1,
            depth,
            prev),
          kernel(kernel), kernelArea(sqr(kernel)), stride(stride), padding(padding),
          alpha(alpha), momentum(momentum), decay(decay),
          inWidth(prev->width), inHeight(prev->height), inDepth(prev->depth),
          weightFiller(weightFiller), biasFiller(biasFiller) {

        weight.resize(kernelArea * inDepth * depth);
        weightHistory.resize(kernelArea * inDepth * depth);
        weightFiller->fill(&weight);

        delta.resize(kernelArea * inDepth * depth);

        bias.resize(depth * height * width);
        biasHistory.resize(depth * height * width);

        biasMultiplier = Vec(height * width, 1.0);

        biasDelta.resize(depth * height * width);
        biasFiller->fill(&bias);

        reshape(num, width, height, depth, &constantOutput);

        col.resize(width * height * inDepth * kernelArea);
      }

      const int kernel;
      const int kernelArea;
      const int stride;
      const int padding;

      const Real alpha;
      const Real momentum;
      const Real decay;

      const int inWidth;
      const int inHeight;
      const int inDepth;

      Filler *weightFiller;
      Filler *biasFiller;

      Vec weight;
      Vec delta;
      Vec weightHistory;
      Vec bias;
      // (1, width * height) ones matrix.
      Vec biasDelta;
      Vec biasHistory;
      Vec biasMultiplier;

      Vec col;

      vector<Vec> constantOutput;

      void forward() {
        for (int n = 0; n < num; n++) {
          forwardOnce(prev->output[n], &output[n]);
        }
      }

      void forwardOnce(const Vec &input, Vec *output) {
        im2col(input, inDepth, inHeight, inWidth, kernel, padding, stride, &col);

        gemm(
          CblasNoTrans, CblasNoTrans,
          depth, width * height, kernelArea * inDepth,
          1., weight, col,
          0., output);

        gemm(
          CblasNoTrans, CblasNoTrans,
          depth, width * height, 1,
          1., bias, biasMultiplier,
          1., output);

      }

      void applyUpdate() {
        updateParam(alpha, momentum, decay, &delta, &weight, &weightHistory);
        updateParam(alpha, momentum, decay, &biasDelta, &bias, &biasHistory);
      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&delta);
        clear(&biasDelta);
        clear(&errors);

        for (int n = 0; n < num; n++) {
          backPropOnce(prev->output[n], nextErrors[n], &errors[n]);
        }

        #ifndef TESTING
                applyUpdate();
        #endif
      }

      void backPropOnce(const Vec &input, const Vec &nextErrors, Vec *errors) {
        backPropBias(nextErrors, &biasDelta);
        backPropInput(nextErrors, weight, errors);
        backPropWeight(nextErrors, input, &delta);
      }

      void backPropBias(const Vec &nextErrors, Vec *biasDelta) {
        gemv(
            CblasNoTrans,
            depth, width * height,
            1., nextErrors, biasMultiplier,
            1., biasDelta);
      }

      void backPropInput(const Vec &nextErrors, const Vec &weight, Vec *errors) {
        if (name == "conv1") {
          return;
        }

        gemm(
            CblasTrans, CblasNoTrans,
            kernelArea * inDepth, width * height, depth,
            1., weight, nextErrors,
            0., &col);

        col2im(col, inDepth, inHeight, inWidth, kernel, padding, stride, errors);
      }

      void backPropWeight(const Vec &nextErrors, const Vec &input, Vec *delta) {
        im2col(input, inDepth, inHeight, inWidth, kernel, padding, stride, &col);

        gemm(
            CblasNoTrans, CblasTrans,
            depth, kernelArea * inDepth, width * height,
            1., nextErrors, col,
            1., delta);
      }
  };
}
