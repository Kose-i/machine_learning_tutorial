import numpy as np
import matplotlib.pyplot as plt

ave = 0  # 平均
std = 1  # 標準偏差

def gauss(x):
    return (1/np.sqrt(2*np.pi*pow(std,2))*np.exp(-pow(x-ave,2)/pow(std,2)))
#filename="img/gauss.png"


x = range(0, 10, 1)
x = np.array(x)*0.1

plt.plot(x,gauss(x))
#plt.savefig(filename)
plt.show()
