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
    const double learning_rate;
  public:
    OR_gate():training_data(4,std::vector<int>(2)), training_answer(4), weight(3), learning_rate(0.5) {
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
      weight[0] = 3.0;
      weight[1] = 2.0;
      weight[2] = 4.0;
    }
    double predict(const std::vector<int>& dataTest) const {
      auto z = weight[0]*dataTest[0]+weight[1]*dataTest[1]+weight[2];
      return sigmoid(z);
    }
    void print() {
      for (auto i = 0;i < 4;++i) {
        std::cout << "predict:" << predict(training_data[i]) << "     answer:" << training_answer[i] << '\n';
      }
    }
    double E() const{ // Squared error (except 1/2)
      double sum = 0.;
      for (auto i = 0;i < 4;++i) {
        sum += (training_answer[i]- predict(training_data[i]))*(training_answer[i]- predict(training_data[i]))/2.0;
      }
      return sum;
    }
    void learn() {
      for (auto i = 0;i < 4;++i) {
        auto predict_answer = predict(training_data[i]);
        auto error = training_answer[i] - predict_answer;
        auto learn_error = sigmoid_df(error);
        weight[0] -= learning_rate*2*training_data[i][0]*learn_error;
        weight[1] -= learning_rate*2*training_data[i][1]*learn_error;
        weight[2] -= learning_rate*2*learn_error;
      }
    }
    static double sigmoid(const double& t) {
      return 1/(1+std::exp(-t));
    }
    static double sigmoid_df(const double& t) {
      double sig_t = sigmoid(t);
      return sig_t*(1.0-sig_t);
    }
};

int main(int argc, char** argv) {
  OR_gate test;
  test.print();
  test.learn();
  test.print();
}
