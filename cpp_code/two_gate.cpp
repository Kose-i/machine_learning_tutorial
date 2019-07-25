#include <iostream>
#include <vector>
#include <cmath>

struct Data_training_pair{
  std::vector<std::vector<int>> training_data;
  std::vector<int> training_answer;
};

class TwoGate{
  protected:
    std::vector<std::vector<int>> training_data;
    std::vector<int> training_answer;
    std::vector<std::vector<double>> weight;
    std::vector<double> weight2;
    const double learning_rate;
  public:
    TwoGate():training_data(4,std::vector<int>(2)), training_answer(4), weight(2, std::vector<double>(3)),weight2(3), learning_rate(0.3) {
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
      weight[0][0] = 0.3;
      weight[0][1] = 0.4;
      weight[0][2] = 0.2;
      weight[1][0] = 0.3;
      weight[1][1] = 0.4;
      weight[1][2] = 0.2;
      weight2[0] = 0.3;
      weight2[1] = 0.4;
      weight2[2] = 0.2;
    }
    double predict(const std::vector<int>& dataTest) const {
      std::vector<double> tmp_output(2,0.0);
      for (auto i = 0;i < 2;++i) {
        tmp_output[i] = weight[i][0]*dataTest[0]+weight[i][1]*dataTest[1]+weight[i][2];
      }
      double answer = dataTest[0]*weight2[0]+dataTest[1]*weight2[1]+weight2[2];
      if (answer < 0) return 0;
      else if (1 < answer) return 1;
      return answer;
    }
    double predict(const std::vector<int>& dataTest, std::vector<double>& output) const {
      for (auto i = 0;i < 2;++i) {
        output[i] = weight[i][0]*dataTest[0]+weight[i][1]*dataTest[1]+weight[i][2];
      }
      double answer = dataTest[0]*weight2[0]+dataTest[1]*weight2[1]+weight2[2];
      if (answer < 0) return 0;
      else if (1 < answer) return 1;
      return answer;
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
        weight2[0] -= learning_rate*predict_error[i]*training_data[i][0];
      }
      for (auto i = 0;i < 4;++i) {
        weight2[1] -= learning_rate*predict_error[i]*training_data[i][1];
      }
      for (auto i = 0;i < 4;++i) {
        weight2[2] -= learning_rate*predict_error[i];
      }
      for (auto i = 0;i < 4;++i) {
        weight[0][0] -= learning_rate*predict_error[i]*training_data[i][0];
      }
      for (auto i = 0;i < 4;++i) {
        weight[0][1] -= learning_rate*predict_error[i]*training_data[i][1];
      }
      for (auto i = 0;i < 4;++i) {
        weight[0][2] -= learning_rate*predict_error[i];
      }
    }
};

class XOR_gate: public TwoGate
{
  public:
    XOR_gate(){
      training_data[0] = {0,0};
      training_answer[0] = {0};
      training_data[1] = {0,1};
      training_answer[1] = {1};
      training_data[2] = {1,0};
      training_answer[2] = {1};
      training_data[3] = {1,1};
      training_answer[3] = {0};
      initial_weight();
    }
};

int main(int argc, char** argv) {
  XOR_gate test;
  test.print();
  for (auto i = 0;i < 30;++i) {
    test.learn();
    std::cout << "now_exam:" << i << ", Error:" << test.E() << '\n';
  }
  test.print();
}
