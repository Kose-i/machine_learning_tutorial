import numpy as np
import matplotlib.pyplot as plt

ave = 0.3

def pdf(x, ave):
    print("pow(ave, x):", pow(ave, x))
    print("pow(1-ave, 1-x):", pow((1.0-ave), (1.0-x)))
    return pow(ave, x)*pow((1.0-ave), (1.0-x))

x = range(0, 11, 1)
x = np.array(x)*0.1

y = pdf(x, ave)
print(pdf(2,0.5))
print(y)
plt.plot(x,y)
plt.show()
