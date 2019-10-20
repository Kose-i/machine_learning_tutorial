import numpy as np
import matplotlib.pyplot as plt
from math import factorial

M = 10 # 全体の試行回数
ave = 0.3

def comb(n, r): # nCr
    return factorial(n) / factorial(r) / factorial(n - r)
def binomial(x):
    return comb(M, x)*(pow(ave, x))*(pow(1-ave, M-x))

x = range(0,10,1)
x = np.array(x)

plt.plot(x, binomial(x))
plt.show()
