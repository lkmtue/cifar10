#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "conv_layer.h"
#include "data.h"
#include "fully_connected_layer.h"
#include "input_layer.h"
#include "layer.h"
#include "max_pooling_layer.h"
#include "net.h"
#include "util.h"
#include "softmax_loss_layer.h"
#include "relu_layer.h"
#include "average_pooling_layer.h"
#include "activation.h"
#include "filler.h"

using namespace std;
using namespace con;

auto gaussianFiller1 = GaussianFiller(0, 0.0001);
auto gaussianFiller2 = GaussianFiller(0, 0.01);
auto gaussianFiller3 = GaussianFiller(0, 0.1);
auto constantFiller = ConstantFiller(0);

const int batch = 100;

vector<Layer*> layers;
vector<Sample> trainData, testData;

Real lr = 0.001;
Real momentum = 0.9;
Real weightDecay = 0.004;

void getLayers(vector<Layer*> *layers) {

  layers->push_back(new InputLayer("input", batch, 32, 32, 3));

  layers->push_back(new ConvolutionalLayer("conv1", 32, 5, 1, 2, layers->back(), &gaussianFiller1, &constantFiller));
  layers->push_back(new MaxPoolingLayer("pool1", 3, 2, layers->back()));
  layers->push_back(new ReluLayer("relu1", layers->back()));

  layers->push_back(new ConvolutionalLayer("conv2", 32, 5, 1, 2, layers->back(), &gaussianFiller2, &constantFiller));
  layers->push_back(new ReluLayer("relu2", layers->back()));
  layers->push_back(new AveragePoolingLayer("pool2", 3, 2, layers->back()));

  layers->push_back(new ConvolutionalLayer("conv3", 64, 5, 1, 2, layers->back(), &gaussianFiller2, &constantFiller));
  layers->push_back(new ReluLayer("relu3", layers->back()));
  layers->push_back(new AveragePoolingLayer("pool3", 3, 2, layers->back()));

  layers->push_back(new FullyConnectedLayer("fc1", 64, layers->back(), &gaussianFiller3, &constantFiller));
  layers->push_back(new FullyConnectedLayer("fc2", 10, layers->back(), &gaussianFiller3, &constantFiller));

  layers->push_back(new SoftmaxLossLayer("softmax", layers->back()));

  for (int i = 0; i < layers->size(); i++) {
    cout << layers->at(i)->name << " " << layers->at(i)->depth << " " << layers->at(i)->width << " " << layers->at(i)->height << endl;
  }
}

int main() {
  getLayers(&layers);
  readTrain(&trainData);
  readTest(&testData);
  train(batch, layers, trainData, testData, lr, momentum, weightDecay);
  return 0;
}
