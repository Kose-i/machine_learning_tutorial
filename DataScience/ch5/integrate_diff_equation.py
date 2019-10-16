from scipy import integrate
import math
import numpy as np

# int
def calcPi(x):
    return 4/(1+x**2)
print(integrate.quad(calcPi, 0, 1)) # Int_0^1 (4/1+x^2)

from numpy import sin
integrate.quad(sin, 0, math.pi/1)  # Int_0-\pi (sin(x))

def I(n):
    return integrate.dblquad(lambda t, x: np.exp(-x*t)/t**n, 0, np.inf, lambda x: 1, lambda x: np.inf)
print('n=1:', I(1))
print('n=2:', I(2))
print('n=3:', I(3))
print('n=4:', I(4))

# diff
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ローレンツ方程式
def lorenz_func(v,t,p,r,b):
    return [-p*v[0]+p*v[1], -v[0]*v[2]+r*v[0]-v[1],v[0]*v[1]-b*v[2]]
# パラメータの設定
p = 10
r = 28
b = 8/3
v0 = [0.1,0.1,0.1]
t = np.arange(0,100,0.01)
v = odeint(lorenz_func, v0, t, args=(p,r,b))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(v[:,0],v[:,1],v[:,2])
plt.title('Lorenz')
plt.grid(True)
plt.show()

