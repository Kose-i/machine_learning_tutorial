#include <iostream>

void svdcmp(double **a, int m, int n, double w[], double **v);

int main(int argc, char** argv) {
  constexpr int N = 4;
  constexpr int M = 3;

	double** a=(double **) malloc((size_t)((N+1)*sizeof(double*)));
  for (auto i = 0;i < N+1;++i) a[i] = (double*) malloc((size_t)((M+1)*sizeof(double)));
  //double a[N+1][M+1];
  a[1][1] = 1; a[1][2] = 2; a[1][3] = 3;
  a[2][1] = 6; a[2][2] = 4; a[2][3] = 5;
  a[3][1] = 8; a[3][2] = 9; a[3][3] = 7;
  a[4][1] =10; a[4][2] =11; a[4][3] =12;

  double* w = (double*) malloc((size_t)(N+1)*sizeof(double));
	double** v=(double **) malloc((size_t)((N+1)*sizeof(double*)));
  for (auto i = 0;i < N+1;++i) v[i] = (double*) malloc((size_t)((N+1)*sizeof(double)));

  std::cout << "a\n";
  for (auto i = 0;i < N+1;++i) {
    for (auto j = 0;j < M+1;++j) {
      std::cout << ' ' << a[i][j];
    }
    std::cout << '\n';
  }

  svdcmp(a, N, M, w, v);
  std::cout << "after:" << '\n';
  std::cout << "a\n";
  for (auto i = 0;i < N+1;++i) {
    for (auto j = 0;j < M+1;++j) {
      std::cout << ' ' << a[i][j];
    }
    std::cout << '\n';
  }

  std::cout << "v\n";
  for (auto i = 0;i < N+1;++i) {
    for (auto j = 0;j < N+1;++j) {
      std::cout << ' ' << v[i][j];
    }
    std::cout << '\n';
  }

  std::cout << "w\n";
  for (auto i = 0;i < N+1;++i) {
    std::cout << ' ' << w[i];
  }
  std::cout << '\n';
  free(a);
  free(w);
  free(v);
}
