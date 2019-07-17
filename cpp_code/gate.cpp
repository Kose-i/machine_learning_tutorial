#include <iostream>
#include <vector>
#include <cmath>

struct Data_training_pair{
  std::vector<std::vector<int>> training_data;
  std::vector<int> training_answer;
};

class OR_gate{
  private:
    std::vector<std::vector<int>> training_data;
    std::vector<int> training_answer;
    std::vector<double> weight;
  public:
    OR_gate():training_data(4,std::vector<int>(2)), training_answer(4), weight(3) {
      training_data[0] = {0,0};
      training_answer[0] = {0};
      training_data[1] = {0,1};
      training_answer[1] = {1};
      training_data[2] = {1,0};
      training_answer[2] = {1};
      training_data[3] = {1,1};
      training_answer[3] = {1};
      initial_weight();
    }
    void initial_weight() {
      weight[0] = 0.3;
      weight[1] = 0.2;
      weight[2] = 0.2;
    }
    double predict(const std::vector<int>& dataTest) {
      auto z = weight[0]*dataTest[0]+weight[1]*dataTest[1]+weight[2];
      return sigmoid(z);
    }
    void print() {
      for (auto i = 0;i < 4;++i) {
        std::cout << "predict:" << predict(training_data[i]) << "answer:" << training_answer[i] << '\n';
      }
    }
    static double sigmoid(const double& t) {
      return 1/(1+std::exp(-t));
    }
};

int main(int argc, char** argv) {
  OR_gate test;
  test.print();
}
