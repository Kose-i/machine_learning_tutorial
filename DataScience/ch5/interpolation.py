import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,10, num=11, endpoint=True)
y = np.cos(-X**2/5.0)

plt.plot(X,y,'o')
plt.grid(True)
plt.show()

# 線形補間
from scipy import interpolate
f = interpolate.interp1d(X,y,'linear')
plt.plot(X,f(X),'-')
plt.grid(True)
plt.show()

# スプライン3次補間
f2 = interpolate.interp1d(X,y,'cubic')

xnew = np.linspace(0,10,num=30,endpoint=True)

plt.plot(X,y,'o',xnew,f(xnew),'-',xnew,f2(xnew),'--')
plt.grid(True)
plt.show()
