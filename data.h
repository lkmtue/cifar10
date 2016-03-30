#pragma once

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "util.h"

namespace con {

  namespace {
    using std::ifstream;

    Vec mean;
  }

  struct Sample {
    Sample() {}

    Vec input;

    int label;
  };

  void computeMean(const vector<Sample> &data, Vec *mean) {
    if (!data.size()) {
      return;
    }

    mean->clear();
    mean->resize(data[0].input.size());

    for (int i = 0; i < mean->size(); i++) {
      mean->at(i) = 0;
      for (int j = 0; j < data.size(); j++) {
        mean->at(i) += data[j].input[i];
      }
      mean->at(i) /= data.size();
    }
  }

  void shuffle(vector<Sample> *data) {
    for (int i = 0; i < data->size(); i++) {
      int j = rand() % (i + 1);
      std::swap(data->at(i), data->at(j));
    }
  }

  void refineMean(const Vec &mean, Vec *input) {
    for (int j = 0; j < input->size(); j++) {
      input->at(j) -= mean[j];
    }
  }

  void refineMean(const Vec &mean, vector<Sample> *data) {
    for (int i = 0; i < data->size(); i++) {
      refineMean(mean, &data->at(i).input);
    }
  }

  void readFrom(const string &fileName, vector<Sample> *data) {
    cout << "Reading from " << fileName << endl;
    ifstream input(fileName.c_str(), ifstream::binary);

    if (input.is_open()) {
      char *mem = new char[3073];

      for (int i = 0; i < 10000; i++) {
        if (i % 1000 == 0) {
          cout << "Sample " << i << endl;
        }

        data->push_back(Sample());

        input.read(mem, 3073);

        data->back().label = mem[0];

        for (int j = 1; j <= 3072; j++) {
          double x = (double)(unsigned char)mem[j];
          data->back().input.push_back(x);
        }
      }

      delete[] mem;
    }
  }

  void readTrain(vector<Sample> *data) {
    data->clear();

    for (char c = '1'; c <= '5'; c++) {
      string fileName = "data/data_batch_";
      fileName += c;
      fileName += ".bin";
      readFrom(fileName, data);
    }

    cout << "Computing mean" << endl;
    computeMean(*data, &mean);

    cout << "Refining mean" << endl;
    refineMean(mean, data);

    cout << "Shuffling" << endl;
    shuffle(data);
  }

  void readTest(vector<Sample> *data) {
    data->clear();
    readFrom("data/test_batch.bin", data);
    refineMean(mean, data);
  }

  void printOutput(const vector<int> &result) {
    std::ofstream output("output", std::ofstream::out);

    output << "id,label" << std::endl;

    vector<std::string> label = {
      "airplane",
      "automobile",
      "bird",
      "cat",
      "deer",
      "dog",
      "frog",
      "horse",
      "ship",
      "truck"
    };

    for (int i = 0; i < result.size(); i++) {
      std::string line = std::to_string(i + 1) + "," + label[result[i]];
      output << line << std::endl;
    }
  }

}
