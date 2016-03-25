#pragma once

#include <vector>

#include "util.h"

namespace con {
  inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
  }

  void im2col(
      const Vec &input,
      const int &inDepth, const int &inHeight, const int &inWidth,
      const int &kernel, const int &padding, const int &stride,
      Vec *output) {

    const Real *data_im = &input[0];
    Real *data_col = &output->at(0);

    const int dilation_h = 1;
    const int dilation_w = 1;

    const int kernel_w = kernel;
    const int kernel_h = kernel;

    const int pad_h = padding;
    const int pad_w = padding;

    const int stride_h = stride;
    const int stride_w = stride;

    const int channels = inDepth;
    const int height = inHeight;
    const int width = inWidth;

    const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  }

  void col2im(
      const Vec &dataCol,
      const int &inDepth, const int &inHeight, const int &inWidth,
      const int &kernel, const int &padding, const int &stride,
      Vec *dataIm) {

    if (dataIm->size() != inHeight * inWidth * inDepth) {
      dataIm->resize(inHeight * inWidth * inDepth);
    }
    clear(dataIm);

    const Real *data_col = &dataCol[0];
    Real *data_im = &dataIm->at(0);

    const int kernel_w = kernel;
    const int kernel_h = kernel;
    const int pad_h = padding;
    const int pad_w = padding;
    const int stride_h = stride;
    const int stride_w = stride;
    const int dilation_h = 1;
    const int dilation_w = 1;

    const int channels = inDepth;
    const int height = inHeight;
    const int width = inWidth;

    const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              data_col += output_w;
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  data_im[input_row * width + input_col] += *data_col;
                }
                data_col++;
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  }

}
