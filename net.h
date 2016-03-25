#pragma once

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include "data.h"
#include "input_layer.h"
#include "layer.h"
#include "util.h"
#include "softmax_loss_layer.h"

namespace con {
  namespace {
    vector<Vec> input;
    vector<int> output;
  }

  void train(const vector<Layer*> &layers, const vector<Vec> &input, const vector<int> &output) {
    InputLayer *inputLayer = (InputLayer*)layers[0];
    SoftmaxLossLayer *outputLayer = (SoftmaxLossLayer*)layers.back();

    inputLayer->setOutput(input);
    outputLayer->setLabels(output);

    for (int l = 0; l < layers.size(); l++) {
      // cout << "Forward: " << layers[l]->name << endl;
      layers[l]->forward();
    }

    for (int l = (int)layers.size() - 1; l >= 0; l--) {
      // cout << "Backward: " << layers[l]->name << endl;
      if (l + 1 < layers.size()) {
        layers[l]->backProp(layers[l + 1]->errors);
      } else {
        layers[l]->backProp(vector<Vec>());
      }
    }
  }

  void train(const int &batchSize, const vector<Layer*> &layers, const vector<Sample> &trainData) {
    for (int times = 0; times < 10; times++) {
      for (int i = 0; i < trainData.size(); i += batchSize) {

        int j = std::min((int)trainData.size(), i + batchSize);

        input.clear();
        output.clear();

        for (int k = i; k < j; k++) {
          input.push_back(trainData[k].input);
          output.push_back(trainData[k].label);
        }

        train(layers, input, output);
      }
    }
  }

  void test(const vector<Layer*> &layers, const vector<Vec> &inputs, vector<int> *results) {
    InputLayer *inputLayer = (InputLayer*)layers[0];
    SoftmaxLossLayer *outputLayer = (SoftmaxLossLayer*)layers.back();

    inputLayer->setOutput(inputs);

    for (int l = 0; l < layers.size(); l++) {
      layers[l]->forward();
    }

    outputLayer->getResults(results);
  }
}
