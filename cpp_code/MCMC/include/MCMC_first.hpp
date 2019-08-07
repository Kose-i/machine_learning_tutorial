#ifndef MCMC_first_cpp
#define MCMC_first_cpp

#include <vector>

struct MCMC_Node{
  public:
    double get_reward()const;
    void set_reward(const double&);
    void add_path(MCMC_Node&);
    std::vector<MCMC_Node*> path;
  private:
    double reward;
};

class MCMC_first{
  private:
    MCMC_Node root_node;
  public:
    MCMC_first();
    MCMC_Node forward();
};
#endif
