import numpy as np
import matplotlib.pyplot as plt

filename="img/gauss.png"

ave = 0  # 平均
std = 1  # 標準偏差

x = np.random.normal(ave, std, 1000)

plt.hist(x, bins=50)
plt.savefig(filename)
plt.show()
