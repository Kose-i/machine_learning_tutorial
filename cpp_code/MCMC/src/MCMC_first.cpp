
#include "MCMC_first.hpp"

#include <vector>

double MCMC_Node::get_reward() const{
  return reward;
};
void MCMC_Node::set_reward(const double& reward_) {
  this->reward = reward_;
};
void MCMC_Node::add_path(MCMC_Node& t) {
  path.push_back(&t);
};

MCMC_first::MCMC_first() {};
MCMC_Node MCMC_first::forward() {};
