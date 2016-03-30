#pragma once

#include <string>
#include <vector>

#include <boost/random.hpp>
#include <cblas.h>

namespace con {

  using std::vector;
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::string;

  static boost::mt19937 rng(0);

  #define BUG(x) std::cout<<#x<<" = "<<(x)<<std::endl;

  typedef double Real;

  typedef vector<Real> Vec;

  Real sigmoid(const Real &z) {
    return 1.0 / (1.0 + std::exp(-z));
  }

  Real derivativeSigmoid(const Real &v) {
    return v * (1.0 - v);
  }

  Real sqr(const Real &x) {
    return x * x;
  }

  Real randomize(const Real &min, const Real &max) {
    boost::uniform_real<Real> dst(min, max);
    return dst(rng);
  }

  void randomizeVec(const Real &min, const Real &max, Vec *a) {
    for (auto it = a->begin(); it != a->end(); it++) {
      (*it) = randomize(min, max);
    }
  }

  void gaussianRng(const Real &mean, const Real &std, Vec *a) {
    boost::normal_distribution<Real> random_distribution(mean, std);
    boost::variate_generator<boost::mt19937, boost::normal_distribution<Real>>
      variate_generator(rng, random_distribution);

    for (auto it = a->begin(); it != a->end(); it++) {
      *it = variate_generator();
    }
  }

  void clear(Vec *a) {
    for (auto it = a->begin(); it != a->end(); it++) {
      (*it) = 0;
    }
  }

  void clear(vector<Vec> *a) {
    for (int i = 0; i < a->size(); i++) {
      clear(&a->at(i));
    }
  }

  void print(const Vec &a) {
    for (auto x : a) {
      cout << x << " ";
    }
    cout << endl;
  }

  void reshape(const int num, const int width, const int height, const int depth, vector<Vec> *a) {
    a->resize(num);
    for (int i = 0; i < num; i++) {
      a->at(i).resize(width * height * depth);
    }
  }

  int ceilDiv(const int &x, const int &y) {
    return (x + y - 1) / y;
  }

  void gemm(
      const CBLAS_TRANSPOSE &TransA, const CBLAS_TRANSPOSE &TransB,
      const int &M, const int &N, const int &K,
      const Real &alpha, const Vec &vecA, const Vec &vecB,
      const Real &beta, Vec *vecC) {

    const Real *A = &vecA[0];
    const Real *B = &vecB[0];
    Real *C = &vecC->at(0);

    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;

    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
        ldb, beta, C, N);
  }

  void gemv(
      const CBLAS_TRANSPOSE &TransA,
      const int &M, const int &N,
      const Real &alpha, const Vec &matA, const Vec &vecX,
      const double beta, Vec *vecY) {

    const Real *A = &matA[0];
    const Real *x = &vecX[0];
    Real *y = &vecY->at(0);

    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
  }

  void vexp(const int &n, const Vec &input, Vec *output) {
    // vsExp(n, &input[0], &output->at(0));
    for (int i = 0; i < n; i++) {
      output->at(i) = exp(input[i]);
    }
  }

  void vdiv(const int &n, const Vec &a, const Vec &b, Vec *c) {
    // vsDiv(n, &a[0], &b[0], &c->at(0));
    for (int i = 0; i < n; i++) {
      c->at(i) = a[i] / b[i];
    }
  }

  void copy(const int &n, const Vec &input, Vec *output) {
    for (int i = 0; i < n; i++) {
      output->at(i) = input[i];
    }
  }

  void ones(const int &n, Vec *v) {
    v->resize(n);
    std::fill(v->begin(), v->end(), 1.0);
  }
}
