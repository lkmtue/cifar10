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

  // Return the number of correct prediction.
  int validateSingleBatch(const vector<Layer*> &layers, const vector<Vec> &input, const vector<int> &output) {
    InputLayer *inputLayer = (InputLayer*)layers[0];
    SoftmaxLossLayer *outputLayer = (SoftmaxLossLayer*)layers.back();

    inputLayer->setOutput(input);
    outputLayer->setLabels(output);

    for (int l = 0; l < layers.size(); l++) {
      layers[l]->forward();
    }

    vector<int> results;
    outputLayer->getResults(&results);

    int correct = 0;
    for (int i = 0; i < results.size(); i++) {
      if (results[i] == output[i]) {
        correct++;
      }
    }

    return correct;
  }

  void validate(const int &batchSize, const vector<Layer*> &layers, const vector<Sample> &validateData) {
    int correct = 0;

    for (int i = 0; i < validateData.size(); i += batchSize) {
      if (i % 1000 == 0) {
        cout << "Validating: " << i << endl;
      }

      int j = std::min((int)validateData.size(), i + batchSize);

      input.clear();
      output.clear();

      for (int k = i; k < j; k++) {
        input.push_back(validateData[k].input);
        output.push_back(validateData[k].label);
      }

      correct += validateSingleBatch(layers, input, output);
    }

    cout << "Accuracy: " << 1.0 * correct / validateData.size() << endl;
  }

  void trainSingleBatch(
      const vector<Layer*> &layers,
      const vector<Vec> &input, const vector<int> &output,
      const Real &lr, const Real &momentum, const Real &decay) {

    InputLayer *inputLayer = (InputLayer*)layers[0];
    SoftmaxLossLayer *outputLayer = (SoftmaxLossLayer*)layers.back();

    inputLayer->setOutput(input);
    outputLayer->setLabels(output);

    // Forward.
    for (int l = 0; l < layers.size(); l++) {
      layers[l]->forward();
    }

    cout << "loss: " << outputLayer->l << endl;

    // Back propagation.
    for (int l = (int)layers.size() - 1; l >= 0; l--) {
      if (l + 1 < layers.size()) {
        layers[l]->backProp(layers[l + 1]->errors);
      } else {
        layers[l]->backProp(vector<Vec>());
      }
    }

    // Apply changes.
    for (int l = 0; l < layers.size(); l++) {
      layers[l]->applyUpdate(lr, momentum, decay);
    }
  }

  void train(
      const int &batchSize,
      const vector<Layer*> &layers,
      const vector<Sample> &trainData, const vector<Sample> &validateData,
      const Real &lr, const Real &momentum, const Real &decay) {

    validate(batchSize, layers, validateData);

    for (int epoch = 0; epoch < 10; epoch++) {
      cout << "Start epoch #" << epoch << endl;

      for (int i = 0; i < trainData.size(); i += batchSize) {
        int j = std::min((int)trainData.size(), i + batchSize);

        input.clear();
        output.clear();

        for (int k = i; k < j; k++) {
          input.push_back(trainData[k].input);
          output.push_back(trainData[k].label);
        }

        trainSingleBatch(layers, input, output, lr, momentum, decay);
      }

      cout << "End epoch #" << epoch << endl;

      validate(batchSize, layers, validateData);
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
