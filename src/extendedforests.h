#ifndef __exforests_h
#define __exforests_h
#include "iforests.h"

class ExtendedNode : public AbstractNode{
  
public:
  ExtendedNode(const unsigned* selection, unsigned size, Node* left, Node* right, 
        const double* n, double t, unsigned* idx, unsigned k)
    :AbstractNode(selection, size, left, right){
    normalVector=n;
    pdotn=t;
    this->idx=idx;
    this->k=k;
  }
  
  virtual Node* branch(const double* x, unsigned n);
  virtual ~ExtendedNode(){
    delete normalVector;
    delete idx;
  }
  
private:
  
  const double* normalVector;
  double pdotn;
  const unsigned* idx;
  unsigned k;
};

class ExtendedTreeBuilder : public TreeBuilder{
  
public:
  
  ExtendedTreeBuilder(unsigned el){
    this->extensionLevel=el;
  }
  
  virtual Node* root(const double* x, unsigned m, unsigned* selection, unsigned nsel,
                      double limit, std::default_random_engine& rnd);
  Node* node(const unsigned* selection, unsigned nsel, unsigned level);
  
private:
  
  const double* data;
  unsigned m;
  unsigned limit, extensionLevel;
  std::default_random_engine engine;
};

#endif