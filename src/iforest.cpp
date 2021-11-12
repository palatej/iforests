#include "iforest.h"
#include <random>

using namespace Rcpp;

Node* INode::branch(const double* x, unsigned n){
  double d=x[axis];
  return d<threshold ? left() : right();
}

Node* ITreeBuilder::root(const double* x, unsigned m, 
                          unsigned* selection, unsigned nsel, double limit, 
                          std::default_random_engine& rnd) {
  this->data=x;
  this->m=m;
  this->limit=limit;
  this->engine=rnd;
  return node(selection, nsel, 0);
}

Node* ITreeBuilder::node(const unsigned* items, unsigned size, unsigned level) {
  if (level >= limit || size <= 1) {
    return new FinalNode(size);
  }
  std::uniform_int_distribution<> distrib(0, m-1);
  unsigned a = distrib(engine);
  const double* z=data+a;
  double xmin=z[items[0]*m], xmax=xmin;
  for (unsigned i = 1; i < size; i++) {
    double r = z[items[i]*m];
    if (r<xmin)
      xmin=r;
    else if (r>xmax)
      xmax=r;
  }
  std::uniform_real_distribution<double> u(0,1);
  double p = u(engine) * (xmax - xmin) + xmin;
  

  // Implement splitting criterion 
  unsigned nl=0;
  for (unsigned i = 0; i < size; ++i) {
    double r = z[items[i]*m];
    if (r < p) {
      ++nl;
    }
  }
  unsigned* XL=new unsigned[nl], *XR=new unsigned[size-nl];
  for (unsigned i = 0, il=0, ir=0; i < size; ++i) {
    unsigned cur=items[i];
    double r = z[cur*m];
    if (r < p) {
      XL[il++]=cur;
    } else {
      XR[ir++]=cur;
    }
  }
  
  Node* left = node(XL, nl, level + 1);
  Node* right = node(XR, size-nl, level + 1);
  return new INode(items, size, left, right, a, p);
}

// [[Rcpp::export]]
NumericVector IsolationForest(NumericMatrix M, int K=200, int N=512, double limit=0){
  std::default_random_engine rnd{static_cast<long unsigned>(time(0))};  
  std::vector<double>  v = as< std::vector<double> >(M);
  double* pdata=v.data();
  int m=M.rows(), n=M.cols();
  ITreeBuilder builder;
  Forest forest(limit, builder);
  forest.fit(pdata, m, n, K, N);
  std::vector<double> d=forest.predict(NULL, n);
  NumericVector D=NumericVector(K);
  D.assign(d.begin(), d.end());
  return D;
}

