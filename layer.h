#pragma once

#include "util.h"

namespace con {
  class Layer {
    public:
      Layer(
        const string &name,
        int num, int width, int height, int depth,
        Layer *prev) :
        name(name), num(num), width(width), height(height), depth(depth), prev(prev) {

        reshape(num, width, height, depth, &output);

        if (prev) {
          reshape(prev->num, prev->width, prev->height, prev->depth, &errors);
        }
      }

      virtual void forward() = 0;
      virtual void backProp(const vector<Vec> &nextErrors) = 0;

      const string name;

      const int num;
      const int width;
      const int height;
      const int depth;

      Layer *prev;

      vector<Vec> output;
      vector<Vec> errors;

    protected:
      int getIndex(const int &d, const int &h, const int &w) {
        return d * (height * width) + h * width + w;
      }
  };
}
