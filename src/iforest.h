#ifndef __iforest_h
#define __iforest_h
#include "iforests.h"

class INode : public AbstractNode{
  
public:
  INode(const unsigned* selection, unsigned size, Node* left, Node* right, 
        unsigned a, double t)
    :AbstractNode(selection, size, left, right){
    axis=a;
    threshold=t;
  }
  
  virtual Node* branch(const double* x, unsigned n);
  virtual ~INode(){
  }
  
private:
  
  unsigned axis;
  double threshold;
  
};

class ITreeBuilder : public TreeBuilder{
  
public:
  
  virtual Node* root(const double* x, unsigned m, unsigned* selection, unsigned nsel,
                      double limit, std::default_random_engine& rnd);
  Node* node(const unsigned* selection, unsigned nsel, unsigned level);
  
private:
  
  const double* data;
  unsigned m;
  unsigned limit;
  std::default_random_engine engine;
};


#endif
