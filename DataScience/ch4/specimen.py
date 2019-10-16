import numpy as np
import matplotlib.pyplot as plt

# カイ二乗分布
# 自由度2, 10, 60に従うカイ二乗分布が生成する乱数のヒストグラム
for df, c in zip([2,10,60], 'bgr'):
    x = np.random.chisquare(df, 1000)
    plt.hist(x,20, color=c)
    plt.grid(True)
    plt.show()

# t分布
x = np.random.standard_t(5,1000)
plt.hist(x)
plt.grid(True)
plt.show()

# F分布
for df, c in zip([(6,7), (10,10), (20,25)], 'bgr'):
    x = np.random.f(df[0], df[1], 1000)
    plt.hist(x,100,color=c)
    plt.grid(True)
    plt.show()
