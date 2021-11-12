#ifndef __xforests_h
#define __xforests_h
#include "iforests.h"

class XNode : public AbstractNode{

public:
  XNode(const unsigned* selection, unsigned size, Node* left, Node* right, 
        const double* n, double t)
    :AbstractNode(selection, size, left, right){
    normalVector=n;
    threshold=t;
  }
  
  virtual Node* branch(const double* x, unsigned n);
  virtual ~XNode(){
      delete normalVector;
  }
  
private:
  
  const double* normalVector;
  double threshold;

};

class XTreeBuilder : public TreeBuilder{
  
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