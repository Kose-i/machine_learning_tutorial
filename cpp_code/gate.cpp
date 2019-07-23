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
    OR_gate():training_data(4,std::vector<int>(2)), training_answer(4), weight(3), learning_rate(0.3) {
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
      weight[1] = 0.4;
      weight[2] = 0.2;
    }
    double predict(const std::vector<int>& dataTest) const {
      return weight[0]*dataTest[0]+weight[1]*dataTest[1]+weight[2];
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
      double sum_error = 0.0;
      std::vector<double> predict_answer(4);
      std::vector<double> predict_error(4);
      for (auto i = 0;i < 4;++i) {
        predict_answer[i] = predict(training_data[i]);
        predict_error[i] = predict_answer[i] - training_answer[i];
        sum_error += learning_rate*predict_answer[i]-training_answer[i];
      }
      for (auto i = 0;i < 4;++i) {
        weight[0] -= learning_rate*predict_error[i]*training_data[i][0];
      }
      for (auto i = 0;i < 4;++i) {
        weight[1] -= learning_rate*predict_error[i]*training_data[i][1];
      }
      for (auto i = 0;i < 4;++i) {
        weight[2] -= learning_rate*predict_error[i];
      }
    }
};

int main(int argc, char** argv) {
  OR_gate test;
  test.print();
  for (auto i = 0;i < 30;++i) {
    test.learn();
    std::cout << "now_exam:" << i << ", Error:" << test.E() << '\n';
  }
  test.print();
}
