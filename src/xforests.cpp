#include "xforests.h"
#include <random>

using namespace Rcpp;

Node* XNode::branch(const double* x, unsigned n){
  double d=IsolationForests::innerProduct(normalVector, x, n);
  return d<threshold ? left() : right();
}

Node* XTreeBuilder::root(const double* x, unsigned m, 
                          unsigned* selection, unsigned nsel, double limit, 
                          std::default_random_engine& rnd) {
  this->data=x;
  this->m=m;
  this->limit=limit;
  this->engine=rnd;
  return node(selection, nsel, 0);
}

Node* XTreeBuilder::node(const unsigned* items, unsigned size, unsigned level) {
  if (level >= limit || size <= 1) {
    return new FinalNode(size);
  }
  std::normal_distribution<double> normal;
  double* nvec = new double[m];
  for (unsigned i = 0; i < m; i++) {
    nvec[i] = normal(engine);
  }
  std::vector<double> d(size);
  double xdotn = IsolationForests::innerProduct(data+m*items[0], nvec, m);
  d[0] = xdotn;
  double min = xdotn, max = xdotn;
  for (unsigned i = 1; i < size; ++i) {
    const double* col=data+m*items[i];
    xdotn = IsolationForests::innerProduct(col, nvec, m);
    d[i] = xdotn;
    if (xdotn < min) {
      min = xdotn;
    } else if (xdotn > max) {
      max = xdotn;
    }
  }
  
  // Implement splitting criterion 
  std::uniform_real_distribution<double> u(0,1);
  double p = u(engine) * (max - min) + min;
  unsigned nl=0;
  for (unsigned i = 0; i < size; ++i) {
    if (d[i] < p) {
      ++nl;
    }
  }
  unsigned* XL=new unsigned[nl], *XR=new unsigned[size-nl];
  for (unsigned i = 0, il=0, ir=0; i < size; ++i) {
    if (d[i] < p) {
      XL[il++]=items[i];
    } else {
      XR[ir++]=items[i];
    }
  }
  
  Node* left = node(XL, nl, level + 1);
  Node* right = node(XR, size-nl, level + 1);
  return new XNode(items, size, left, right, nvec, p);
}

// [[Rcpp::export]]
NumericVector XIsolationForest(NumericMatrix M, int K=200, int N=512, double limit=0){
  std::default_random_engine rnd{static_cast<long unsigned>(time(0))};  
  std::vector<double>  v = as< std::vector<double> >(M);
  double* pdata=v.data();
  int m=M.rows(), n=M.cols();
  XTreeBuilder builder;
  Forest forest(limit, builder);
  forest.fit(pdata, m, n, K, N);
  std::vector<double> d=forest.predict(NULL, n);
  NumericVector D=NumericVector(K);
  D.assign(d.begin(), d.end());
  return D;
}

