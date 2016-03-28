#define TESTING 1

#include "input_layer.h"
#include "util.h"
#include "softmax_loss_layer.h"
#include "fully_connected_layer.h"
#include "filler.h"
#include "activation.h"
#include "conv_layer.h"
#include "max_pooling_layer.h"
#include "average_pooling_layer.h"
#include "relu_layer.h"

namespace con {

  #define eps 1e-6

  InputLayer generateInput(const Real &min, const Real &max, const int &num, const int &width, const int &height, const int &depth) {
    InputLayer inputLayer("input", num, width, height, depth);

    vector<Vec> output;
    output.resize(num);
    for (int i = 0; i < num; i++) {
      output[i].resize(width * height * depth);
      randomizeVec(min, max, &output[i]);
    }

    inputLayer.setOutput(output);

    return inputLayer;
  }

  vector<int> generateLabels(const int &num, const int &depth) {
    vector<int> labels;
    for (int i = 0; i < num; i++) {
      labels.push_back(rand() % depth);
    }
    return labels;
  }

  void testSoftmax() {
    int num = 100;
    int width = 1;
    int height = 1;
    int depth = 10;
    InputLayer inputLayer = generateInput(-2, 2, num, width, height, depth);
    vector<int> labels = generateLabels(num, depth);
    SoftmaxLossLayer softmaxLayer("softmax", &inputLayer);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();
    softmaxLayer.backProp(vector<Vec>());
    const vector<Vec> &errors = softmaxLayer.errors;

    for (int n = 0; n < softmaxLayer.num; n++) {
      for (int i = 0; i < softmaxLayer.depth; i++) {
        InputLayer inputPlusEps = inputLayer;
        inputPlusEps.output[n][i] += eps;
        SoftmaxLossLayer softmaxPlusEps("softmax", &inputPlusEps);
        softmaxPlusEps.setLabels(labels);
        Real lossPlusEps = softmaxPlusEps.loss();

        InputLayer inputMinusEps = inputLayer;
        inputMinusEps.output[n][i] -= eps;
        SoftmaxLossLayer softmaxMinusEps("softmax", &inputMinusEps);
        softmaxMinusEps.setLabels(labels);
        Real lossMinusEps = softmaxMinusEps.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = errors[n][i];

        cout << estimated << " " << calculated << " " << fabs(estimated - calculated) << endl;
        if (fabs(estimated - calculated) > eps) {
          cout << "ERROR!" << endl;
          exit(0);
        }
      }
    }
  }

  void testFC() {
    int num = 10;
    int width = 32;
    int height = 32;
    int depth = 3;
    auto uniform2 = UniformFiller(-2, 2);
    auto constantActivation = ConstantActivation();

    InputLayer inputLayer = generateInput(0, 0.01, num, width, height, depth);

    FullyConnectedLayer fcLayer("fc", 10, 1, 0, &inputLayer, &uniform2, &uniform2, &constantActivation);
    fcLayer.forward();

    SoftmaxLossLayer softmaxLayer("softmax", &fcLayer);
    vector<int> labels = generateLabels(num, 10);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();

    softmaxLayer.backProp(vector<Vec>());
    fcLayer.backProp(softmaxLayer.errors);

    for (int i = 0; i < fcLayer.weight.size(); i++) {
      Real save = fcLayer.weight[i];
      fcLayer.weight[i] = save + eps;
      fcLayer.forward();

      Real lossPlusEps = softmaxLayer.loss();

      fcLayer.weight[i] = save - eps;
      fcLayer.forward();
      Real lossMinusEps = softmaxLayer.loss();

      Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
      Real calculated = fcLayer.delta[i];

      cout << estimated << " " << calculated << " " << estimated - calculated << endl;

      if (std::fabs(estimated - calculated) > 1e-8) {
        cout << "ERROR!" << endl;
        exit(0);
      }

      fcLayer.weight[i] = save;
    }

    for (int i = 0; i < fcLayer.bias.size(); i++) {
      Real save = fcLayer.bias[i];
      fcLayer.bias[i] = save + eps;
      fcLayer.forward();

      Real lossPlusEps = softmaxLayer.loss();

      fcLayer.bias[i] = save - eps;
      fcLayer.forward();
      Real lossMinusEps = softmaxLayer.loss();

      Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
      Real calculated = fcLayer.biasDelta[i];

      cout << "bias " << i << " " << estimated << " " << calculated << " " << estimated - calculated << endl;

      if (std::fabs(estimated - calculated) > 1e-8) {
        cout << "ERROR!" << endl;
        exit(0);
      }

      fcLayer.bias[i] = save;
    }

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < inputLayer.output.size(); i++) {
        Real save = inputLayer.output[n][i];
        inputLayer.output[n][i] = save + eps;
        fcLayer.forward();
        Real lossPlusEps = softmaxLayer.loss();

        inputLayer.output[n][i] = save - eps;
        fcLayer.forward();
        Real lossMinusEps = softmaxLayer.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = fcLayer.errors[n][i];

        cout << estimated << " " << calculated << " " << estimated - calculated << endl;

        if (std::fabs(estimated - calculated) > 1e-8) {
          cout << "ERROR!" << endl;
          exit(0);
        }

        inputLayer.output[n][i] = save;
      }
    }
  }

  void testConv() {
    int num = 2;
    int width = 32;
    int height = 32;
    int depth = 3;
    auto uniform1 = UniformFiller(-1, 1);
    auto uniform2 = UniformFiller(-2, 2);
    auto constantActivation = ConstantActivation();

    InputLayer inputLayer = generateInput(0, 0.1, num, width, height, depth);

    ConvolutionalLayer convLayer("conv", 32, 5, 1, 2, 1, 0, &inputLayer, &uniform1, &uniform1);
    convLayer.forward();

    FullyConnectedLayer fcLayer("fc", 10, 1, 0, &convLayer, &uniform2, &uniform2, &constantActivation);

    fcLayer.forward();

    SoftmaxLossLayer softmaxLayer("softmax", &fcLayer);
    vector<int> labels = generateLabels(num, 10);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();

    softmaxLayer.backProp(vector<Vec>());
    fcLayer.backProp(softmaxLayer.errors);
    convLayer.backProp(fcLayer.errors);

    for (int i = 0; i < convLayer.weight.size(); i++) {
      Real save = convLayer.weight[i];

      convLayer.weight[i] = save + eps;
      convLayer.forward();
      fcLayer.forward();
      Real lossPlusEps = softmaxLayer.loss();

      convLayer.weight[i] = save - eps;
      convLayer.forward();
      fcLayer.forward();
      Real lossMinusEps = softmaxLayer.loss();

      Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
      Real calculated = convLayer.delta[i];

      cout << estimated << " " << calculated << " " << estimated - calculated << endl;

      if (std::fabs(estimated - calculated) > 1e-5) {
        cout << "ERROR!" << endl;
        exit(0);
      }

      convLayer.weight[i] = save;
    }

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < inputLayer.output[n].size(); i++) {
        Real save = inputLayer.output[n][i];

        inputLayer.output[n][i] = save + eps;
        convLayer.forward();
        fcLayer.forward();
        Real lossPlusEps = softmaxLayer.loss();

        inputLayer.output[n][i] = save - eps;
        convLayer.forward();
        fcLayer.forward();
        Real lossMinusEps = softmaxLayer.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = convLayer.errors[n][i];

        cout << estimated << " " << calculated << " " << estimated - calculated << endl;

        if (std::fabs(estimated - calculated) > 1e-5) {
          cout << "ERROR!" << endl;
          exit(0);
        }

        inputLayer.output[n][i] = save;
      }
    }

    for (int i = 0; i < convLayer.bias.size(); i++) {
      Real save = convLayer.bias[i];
      convLayer.bias[i] = save + eps;
      convLayer.forward();
      fcLayer.forward();
      Real lossPlusEps = softmaxLayer.loss();

      convLayer.bias[i] = save - eps;
      convLayer.forward();
      fcLayer.forward();
      Real lossMinusEps = softmaxLayer.loss();

      Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
      Real calculated = convLayer.biasDelta[i];

      cout << estimated << " " << calculated << " " << estimated - calculated << endl;

      if (std::fabs(estimated - calculated) > 1e-5) {
        cout << "ERROR!" << endl;
        exit(0);
      }

      convLayer.bias[i] = save;
    }
  }

  void testMaxPooling() {
    int num = 5;
    int width = 10;
    int height = 10;
    int depth = 3;
    auto uniform1 = UniformFiller(-1, 1);
    auto uniform2 = UniformFiller(-2, 2);
    auto constantActivation = ConstantActivation();

    InputLayer inputLayer = generateInput(0, 0.1, num, width, height, depth);

    MaxPoolingLayer maxLayer("max", 3, 1, &inputLayer);
    maxLayer.forward();

    FullyConnectedLayer fcLayer("fc", 10, 1, 0, &maxLayer, &uniform2, &uniform2, &constantActivation);
    fcLayer.forward();

    SoftmaxLossLayer softmaxLayer("softmax", &fcLayer);
    vector<int> labels = generateLabels(num, 10);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();

    softmaxLayer.backProp(vector<Vec>());
    fcLayer.backProp(softmaxLayer.errors);
    maxLayer.backProp(fcLayer.errors);

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < inputLayer.output[n].size(); i++) {
        Real save = inputLayer.output[n][i];

        inputLayer.output[n][i] = save + eps;
        maxLayer.forward();
        fcLayer.forward();
        Real lossPlusEps = softmaxLayer.loss();

        inputLayer.output[n][i] = save - eps;
        maxLayer.forward();
        fcLayer.forward();
        Real lossMinusEps = softmaxLayer.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = maxLayer.errors[n][i];

        BUG(lossMinusEps)
        BUG(lossPlusEps)

        cout << n << " " << i << " " << estimated << " " << calculated << " " << estimated - calculated << endl;

        if (std::fabs(estimated - calculated) > 1e-5) {
          cout << "ERROR!" << endl;
          exit(0);
        }

        inputLayer.output[n][i] = save;
      }
    }
  }

  void testAveragePooling() {
    int num = 5;
    int width = 16;
    int height = 16;
    int depth = 32;
    auto uniform1 = UniformFiller(-1, 1);
    auto uniform2 = UniformFiller(-2, 2);
    auto constantActivation = ConstantActivation();

    InputLayer inputLayer = generateInput(0, 0.1, num, width, height, depth);

    AveragePoolingLayer averageLayer("average", 3, 2, &inputLayer);

    averageLayer.forward();

    FullyConnectedLayer fcLayer("fc", 10, 1, 0, &averageLayer, &uniform2, &uniform2, &constantActivation);
    fcLayer.forward();

    SoftmaxLossLayer softmaxLayer("softmax", &fcLayer);
    vector<int> labels = generateLabels(num, 10);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();

    softmaxLayer.backProp(vector<Vec>());
    fcLayer.backProp(softmaxLayer.errors);
    averageLayer.backProp(fcLayer.errors);

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < inputLayer.output[n].size(); i++) {
        Real save = inputLayer.output[n][i];

        inputLayer.output[n][i] = save + eps;
        averageLayer.forward();
        fcLayer.forward();
        Real lossPlusEps = softmaxLayer.loss();

        inputLayer.output[n][i] = save - eps;
        averageLayer.forward();
        fcLayer.forward();
        Real lossMinusEps = softmaxLayer.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = averageLayer.errors[n][i];

        cout << estimated << " " << calculated << " " << estimated - calculated << endl;

        if (std::fabs(estimated - calculated) > 1e-5) {
          cout << "ERROR!" << endl;
          exit(0);
        }

        inputLayer.output[n][i] = save;
      }
    }
  }

  void testRelu() {
    int num = 5;
    int width = 10;
    int height = 10;
    int depth = 3;
    auto uniform2 = UniformFiller(-2, 2);
    auto constantActivation = ConstantActivation();

    InputLayer inputLayer = generateInput(-1, 1, num, width, height, depth);

    ReluLayer reluLayer("relu", &inputLayer);
    reluLayer.forward();

    FullyConnectedLayer fcLayer("fc", 10, 1, 0, &reluLayer, &uniform2, &uniform2, &constantActivation);
    fcLayer.forward();

    SoftmaxLossLayer softmaxLayer("softmax", &fcLayer);
    vector<int> labels = generateLabels(num, 10);
    softmaxLayer.setLabels(labels);
    softmaxLayer.forward();

    softmaxLayer.backProp(vector<Vec>());
    fcLayer.backProp(softmaxLayer.errors);
    reluLayer.backProp(fcLayer.errors);

    for (int n = 0; n < num; n++) {
      for (int i = 0; i < inputLayer.output[n].size(); i++) {

        Real save = inputLayer.output[n][i];

        inputLayer.output[n][i] = save + eps;
        reluLayer.forward();
        fcLayer.forward();
        Real lossPlusEps = softmaxLayer.loss();

        inputLayer.output[n][i] = save - eps;
        reluLayer.forward();
        fcLayer.forward();
        Real lossMinusEps = softmaxLayer.loss();

        Real estimated = (lossPlusEps - lossMinusEps) / (2 * eps);
        Real calculated = reluLayer.errors[n][i];

        cout << estimated << " " << calculated << " " << estimated - calculated << endl;

        if (std::fabs(estimated - calculated) > 1e-5) {
          cout << "ERROR!" << endl;
          exit(0);
        }

        inputLayer.output[n][i] = save;
      }
    }

  }
}

int main() {
  // con::testSoftmax();
  con::testFC();
  // con::testConv();
  // con::testMaxPooling();
  // con::testAveragePooling();
  // con::testRelu();
}
