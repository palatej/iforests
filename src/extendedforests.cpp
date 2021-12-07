#include "extendedforests.h"
#include <random>

using namespace Rcpp;

Node* ExtendedNode::branch(const double* x, unsigned n){
  double d=IsolationForests::innerProduct(normalVector, x, idx, k);
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
  
  double* nvec = new double[m];
  std::normal_distribution<double> normal;
  unsigned k=0;
  unsigned* idx;
  if (extensionLevel > 0 && extensionLevel < m) {
    k=extensionLevel;
    idx = IsolationForests::sampleWithoutReplacement(extensionLevel, m, false, engine);
    for (unsigned i=0; i<m; ++i )
      nvec[i]=0;
    for (unsigned i = 0; i < extensionLevel; ++i) {
      nvec[idx[i]] = normal(engine);
    }
  }
  else{
    k=m;
    idx=new unsigned[m];
    for (unsigned i = 0; i < m; i++) {
      idx[i]=i;
      nvec[i] = normal(engine);
    }
  }
  std::vector<double> p(m);
  std::vector<double> xmin(m);
  std::vector<double> xmax(m);
  for (unsigned i = 0; i < k; ++i) {
    unsigned c=idx[i];
    double cmin = data[items[0]*m+c], cmax=cmin;
    for (unsigned j = 1; j < size; ++j) {
      double cur = data[items[j]*m+c];
      if (cur < cmin) {
        cmin = cur;
      } else if (cur > cmax) {
        cmax = cur;
      }
    }
    xmin[c] = cmin;
    xmax[c] = cmax;
  }

  
  std::uniform_real_distribution<double> u(0,1);
  for (unsigned i = 0; i < k; i++) {
    unsigned c=idx[i];
    double r = u(engine);
    p[c] = xmin[c] + r * (xmax[c] - xmin[c]);
  }
  
  double pdotn=IsolationForests::innerProduct(nvec, p.data(), idx, k);
  unsigned nl=0;
  std::vector<double> ds(size);
  for (unsigned i = 0; i < size; ++i) {
    double d=IsolationForests::innerProduct(nvec, data+items[i]*m, idx, k);
    ds[i]=d;
    if (d < pdotn) {
      ++nl;
    }
  }
    unsigned* XL=new unsigned[nl], *XR=new unsigned[size-nl];
    for (unsigned i = 0, il=0, ir=0; i < size; ++i) {
      double d=ds[i];
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
    return new ExtendedNode(items, size, left, right, nvec, pdotn, idx, k);
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

