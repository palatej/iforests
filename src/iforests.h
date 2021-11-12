#ifndef iforests_h 
#define iforests_h

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

class Node;

class IsolationForests{
public:
  static double innerProduct(const double* X1, const double* X2, unsigned m) ;
  
  static double cFactor(unsigned N);
  
  static double findPath(Node* node, const double* x, unsigned m);

  static unsigned* sampleWithoutReplacement(unsigned k, unsigned N, bool shuffle, std::default_random_engine& rnd);
  
};

/**
 * The current node is the owner of the selection and of its children and must destroy them itself
 */
class Node{
public:
 
  unsigned size(){return n;}
  virtual Node* right()=0;
  virtual Node* left();

  virtual bool isFinal()=0;
  
  virtual Node* branch(const double* x, unsigned n)=0;
  
  virtual ~Node(){}


protected:
  Node(unsigned size){
    n=size;
  }
  
private:
  
  unsigned n;
};

class FinalNode: public Node{
public:
  
  FinalNode(unsigned size):
  Node(size)
  {
  }

  Node* right(){return NULL;}
  Node* left(){return NULL;}
  
  bool isFinal(){return true;}
  
  virtual Node* branch(const double* x, unsigned n){return NULL;}
  
  virtual ~FinalNode(){
  }

};

class AbstractNode: public Node{
public:
  
  Node* right(){return r;}
  Node* left(){return l;}
  
  bool isFinal(){return false;}
  
  virtual Node* branch(const double* x, unsigned n)=0;
  
  virtual ~AbstractNode(){
    delete l;
    delete r;
    delete sel;
  }
  
protected:
  AbstractNode(const unsigned* selection, unsigned size, Node* left, Node* right)
    : Node(size)
  {
    sel=selection;
    l=left;
    r=right;
  }
  
private:
  
  Node* l, *r; 
  const unsigned* sel;
};


class TreeBuilder{
  
  public:
    
    virtual Node* root(const double* x, unsigned m, 
                       unsigned* selection, unsigned nsel, double limit, 
                       std::default_random_engine& rnd)=0;
  
};

//////////////////

class Forest {
  
public: 
  
  Forest(unsigned limit, TreeBuilder& builder):
  builder(builder){
    this->limit=limit;
 }
  
  ~Forest(){clearTrees();}

  /**
   * Data, dim, size, number of trees, sample size used for each tree
   */
  void fit(const double* x, unsigned m, unsigned n, unsigned ntrees, unsigned sampleSize);
  
  /**
   * Fitted data, dim, size
   */
  std::vector<double> predict(const double* x, unsigned size);
  
    
private:
  
  void clearTrees();
  const double* x;
  unsigned m, n;
  unsigned limit;
  std::vector<Node*> trees;
  double c;
  TreeBuilder& builder;

};



#endif