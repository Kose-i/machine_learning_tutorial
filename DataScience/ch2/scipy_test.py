import scipy.linalg as linalg
import numpy as np

matrix = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])

print("matrix:{0}".format(matrix))
print("matrixの行列式:{0}".format(linalg.det(matrix)))
print("matrixの逆行列:{0}".format(linalg.inv(matrix)))
print("逆行列の確かめ:{0}".format(matrix.dot(linalg.inv(matrix))))

eig_value, eig_vector = linalg.eig(matrix) # 固有値と固有ベクトル
print("固有値:{0}".format(eig_value))
print("固有ベクトル:{0}".format(eig_vector))

#newton Method
#f(x) = x^2 + 2x + 1
def my_func(x):
    return (x**2 + 2*x + 1)

from scipy.optimize import newton
print(newton(my_func, 0))

from scipy.optimize import minimize_scalar
print(minimize_scalar(my_func, method='Brent'))
