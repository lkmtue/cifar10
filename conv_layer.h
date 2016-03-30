#pragma once

#include "filler.h"
#include "im2col.h"
#include "layer.h"
#include "param.h"
#include "util.h"

namespace con {
  class ConvolutionalLayer : public Layer {
    public:
      ConvolutionalLayer(
        const string &name,
        const int &depth, const int &kernel, const int &stride, const int &padding,
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
          weight(kernelArea * inDepth * depth, weightFiller),
          bias(depth * height * width, biasFiller) {

        biasMultiplier = Vec(height * width, 1.0);

        col.resize(width * height * inDepth * kernelArea);
      }

      const int kernel;
      const int kernelArea;
      const int stride;
      const int padding;

      Param weight;
      Param bias;

      // (1, width * height) ones matrix.
      Vec biasMultiplier;

      Vec col;

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
          1., weight.value, col,
          0., output);

        gemm(
          CblasNoTrans, CblasNoTrans,
          depth, width * height, 1,
          1., bias.value, biasMultiplier,
          1., output);

      }

      void backProp(const vector<Vec> &nextErrors) {
        clear(&weight.delta);
        clear(&bias.delta);
        clear(&errors);

        for (int n = 0; n < num; n++) {
          backPropOnce(prev->output[n], nextErrors[n], &errors[n]);
        }
      }

      void backPropOnce(const Vec &input, const Vec &nextErrors, Vec *errors) {
        backPropBias(nextErrors, &bias.delta);
        backPropInput(nextErrors, weight.value, errors);
        backPropWeight(nextErrors, input, &weight.delta);
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

      void applyUpdate(const Real &lr, const Real &momentum, const Real &decay) {
        weight.update(lr, momentum, decay);
        bias.update(lr, momentum, decay);
      }
  };
}
