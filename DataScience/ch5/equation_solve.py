import numpy as np
import matplotlib.pyplot as plt

# optimize
from scipy.optimize import fsolve
def f(x):
    y = 2*x**2 + 2*x - 10
    return y
x = np.linspace(-4,4)
plt.plot(x,f(x))
plt.plot(x,np.zeros(len(x)))
plt.grid(True)
plt.show()

x = fsolve(f,2)
print(x)
x = fsolve(f,-3)
print(x)

from scipy.optimize import minimize
# 目的となる関数
def objective(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return x1*x4*(x1+x2+x3)+x3
# 制約式その1
def constraint1(x):
    return x[0]*x[1]*x[2]*x[3]-25.0 # >=0
# 制約式その2
def constraint2(x):
    sum_sq = 40
    for i in range(4):
        sum_sq = sum_sq - x[i]**2
    return sum_sq #>=0
x0 = [1,5,5,1] # 初期値
print("初期値:",objective(x0))
b = (1.0, 5.0)
bnds = (b,b,b,b)
con1 = {'type':'ineq', 'fun':constraint1}
con2 = {'type':'ineq', 'fun':constraint2}
cons = [con1, con2]
sol = minimize(objective, x0, method='SLSQP',bounds=bnds, constraints=cons)
print('Y:',sol.fun)
print('X:',sol.x)
