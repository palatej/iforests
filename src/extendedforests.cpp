#include "extendedforests.h"
#include <random>

using namespace Rcpp;

Node* ExtendedNode::branch(const double* x, unsigned n){
  double d=IsolationForests::innerProduct(normalVector, x, n);
  return d<pdotn ? left() : right();
}

Node* ExtendedTreeBuilder::root(const double* x, unsigned m, 
                          unsigned* selection, unsigned nsel, double limit,
                          std::default_random_engine& rnd) {
  this->data=x;
  this->m=m;
  this->limit=limit;
  this->engine=rnd;
  return node(selection, nsel, 0);
}

Node* ExtendedTreeBuilder::node(const unsigned* items, unsigned size, unsigned level) {
  if (level >= limit || size <= 1) {
    return new FinalNode(size);
  }
  
  std::normal_distribution<double> normal;
  double* nvec = new double[m];
  for (unsigned i = 0; i < m; i++) {
    nvec[i] = normal(engine);
  }
  std::vector<double> p(m);
  std::vector<double> xmin(m);
  std::vector<double> xmax(m);
  for (unsigned i = 0; i < m; ++i) {
    double cmin = data[items[0]*m+i], cmax=cmin;
    for (unsigned j = 1; j < size; ++j) {
      double cur = data[items[j]*m+i];
      if (cur < cmin) {
        cmin = cur;
      } else if (cur > cmax) {
        cmax = cur;
      }
    }
    xmin[i] = cmin;
    xmax[i] = cmax;
  }

  if (extensionLevel > 0 && extensionLevel < m) {
    int k = m - extensionLevel;
    unsigned* zeroidx = IsolationForests::sampleWithoutReplacement(k, m, false, engine);
    for (int i = 0; i < k; ++i) {
      nvec[zeroidx[i]] = 0;
    }
  }
  
  std::uniform_real_distribution<double> u(0,1);
  for (unsigned i = 0; i < m; i++) {
    double r = u(engine);
    p[i] = xmin[i] + r * (xmax[i] - xmin[i]);
  }
  
 double pdotn=IsolationForests::innerProduct(nvec, p.data(), m);
  unsigned nl=0;
  for (unsigned i = 0; i < size; ++i) {
    double d=IsolationForests::innerProduct(nvec, data+items[i]*m, m);
    if (d < pdotn) {
      ++nl;
    }
  }
    unsigned* XL=new unsigned[nl], *XR=new unsigned[size-nl];
    for (unsigned i = 0, il=0, ir=0; i < size; ++i) {
      double d=IsolationForests::innerProduct(nvec, data+items[i]*m, m);
      if (d < pdotn) {
        XL[il++]=items[i];
      } else {
        XR[ir++]=items[i];
      }
    }
  
    Node* left, *right;
    if (nl <= 1)
      left=new FinalNode(nl);
    else
      left = node(XL, nl, level + 1);
    if (size-nl<=1)
      right=new FinalNode(size-nl);
    else
      right = node(XR, size-nl, level + 1);
    return new ExtendedNode(items, size, left, right, nvec, pdotn);
}

// [[Rcpp::export]]
NumericVector ExtendedIsolationForest(NumericMatrix M, int K=200, int N=512, double limit=0, int extensionLevel=0){
  std::default_random_engine rnd{static_cast<long unsigned>(time(0))};  
  std::vector<double>  v = as< std::vector<double> >(M);
  double* pdata=v.data();
  int m=M.rows(), n=M.cols();
  ExtendedTreeBuilder builder(extensionLevel);
  Forest forest(limit, builder);
  forest.fit(pdata, m, n, K, N);
  std::vector<double> d=forest.predict(NULL, n);
  NumericVector D=NumericVector(K);
  D.assign(d.begin(), d.end());
  return D;
}

