#include "iforests.h"
#include <cstdlib>
#include <chrono>

using namespace Rcpp;

#define EULER_CONSTANT 0.5772156649


double IsolationForests::innerProduct(const double* X1, const double* X2, unsigned n) {
  double result = 0.0;
  for (unsigned i = 0; i < n; i++) {
    result += X1[i] * X2[i];
  }
  return result;
}

double IsolationForests::cFactor(unsigned N) {
  double Nd = (double) N, Ndc = Nd - 1;
  double result;
  result = 2.0 * ((::log(Ndc) + EULER_CONSTANT) - Ndc / Nd);
  return result;
  
}

double IsolationForests::findPath(Node* node, const double* x, unsigned m) {
  if (node->isFinal()) {
    unsigned size = node->size();
    switch (size) {
    case 0:
      return 0; // unused node
    case 1:
      return 1; // true final level
    default:
      return 1 + cFactor(size);
    }
  } else {
    return 1+findPath(node->branch(x, m), x, m);
  }
}

unsigned* IsolationForests::sampleWithoutReplacement(unsigned k, unsigned N, bool shuffle, std::default_random_engine& rnd) {
  
  // Create an unordered set to store the samples
  unsigned* sample=new unsigned[k];
  std::vector<bool> flags(N);

  for (unsigned i = 0, r = N - k; r < N; ++r, ++i) {
    std::uniform_int_distribution<> distrib(0, r);
    unsigned v = distrib(rnd);
    if (flags[v]) {
      flags[r]=1;
      sample[i] = r;
    } else {
      flags[v]=1;
      sample[i] = v;
    }
  }
  
  if (shuffle) // shuffle the results
  {
    std::uniform_int_distribution<> distrib(0, k-1);
    for (unsigned j = 0; j < k; ++j) {
      unsigned idx = distrib(rnd);
      if (idx != j) {
        unsigned tmp = sample[j];
        sample[j] = sample[idx];
        sample[idx] = tmp;
      }
    }
  }
  return sample;
}

void Forest::clearTrees(){
  std::vector<Node*>::iterator cur=trees.begin();
  while (cur != trees.end())
    delete *cur++;
}

void Forest::fit(const double* x, unsigned m, unsigned n, unsigned ntrees, unsigned sampleSize) {
  clearTrees();
  this->x=x;
  this->m=m;
  this->n=n;
  unsigned climit = limit == 0 ? (unsigned) ::ceil(::log(sampleSize) / ::log(2)) : limit;
  c = IsolationForests::cFactor(sampleSize);
  clearTrees();
  trees=std::vector<Node*>(ntrees);
  std::default_random_engine rnd{static_cast<long unsigned>(time(0))};  
  if (sampleSize < n && sampleSize>0) {
    for (unsigned i = 0; i < ntrees; ++i) {
      unsigned* sample = IsolationForests::sampleWithoutReplacement(sampleSize, n, false, rnd);
      trees[i] = builder.root(x, m, sample, sampleSize, climit, rnd);
    }
  } else {
    for (unsigned i = 0; i < ntrees; ++i) {
      unsigned* all=new unsigned[n];
      for (unsigned i=0; i<n; ++i){
        all[i]=i;
      }
      trees[i] = builder.root(x, m, all, n, climit, rnd);
    }
  }
}

std::vector<double> Forest::predict(const double* data, unsigned size) {

  const double* z;
  if (data == NULL){
    z=x;
    size=n;
  }else{
    z=data;
  }

  std::vector<double> S(size);
  const double* cur=z;
  for (unsigned i = 0; i < size; i++, cur+=m) {
       double htemp = 0.0;
       for (unsigned j=0; j<trees.size(); ++j){
          htemp += IsolationForests::findPath(trees[j], cur, m);
       }
       double havg = htemp / trees.size();
       S[i] = ::pow(2.0, -havg / c);
  }
  return S;
}


