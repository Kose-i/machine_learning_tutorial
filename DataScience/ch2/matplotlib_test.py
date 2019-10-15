import matplotlib as mpl
import seaborn as sns
import numpy.random as random

import matplotlib.pyplot as plt
import numpy as np

# 散布図 
random.seed(0)
x = np.random.randn(30)
y = np.sin(x) + np.random.randn(30)
plt.figure(figsize=(20,6)) # グラフの大きさ指定

plt.plot(x,y,'o')
#plt.scatter(x,y)

plt.title('Title Name')
plt.xlabel('X-label')
plt.ylabel('Y-label')

plt.grid(True)
plt.show()

# 連続曲線

numpy_data_x = np.arange(1000)
numpy_random_data_y = np.random.randn(1000).cumsum()

plt.figure(figsize=(20,6))
plt.plot(numpy_data_x, numpy_random_data_y, label='Label')
plt.legend()

plt.xlabel('X-label')
plt.ylabel('Y-label')

plt.grid(True)
plt.show()

# ヒストグラフ
plt.figure(figsize=(20,6))

plt.hist(np.random.randn(10**5)*10+50, bins=60, range=(20,80))
plt.grid(True)

plt.show()
