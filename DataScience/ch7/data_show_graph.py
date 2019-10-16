import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as plt
import seaborn as sns
sns.set()

x = [1,2,3]
y = [10,1,4]

# 棒グラフ
plt.figure(figsize=(10,6))
plt.bar(x,y,align='center', width=0.5)
plt.xticks(x,['A Class','B Class','C Class'])
plt.xlabel('Class')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# 横の棒グラフ
plt.barh(x,y,aligh='center')
plt.yticks(x,['A Class','B Class','C Class'])
plt.ylabel('Class')
plt.xlabel('Score')
plt.grid(True)
plt.show()

# 複数のグラフを描く
y1 = np.array([30,10,40])
y2 = np.array([10,50,90])

x = np.arange(len(y1))
w = 0.4

plt.bar(x,  y1,color='blue', width=w,label='Math first',align='center')
plt.bar(x+w,y2,color='green',width=w,label='Math final',align='center')

plt.legend(loc='best')

plt.xticks(x + w/2, ['Class A','Class B','Class C'])
plt.grid(True)
plt.show()

# 積み上げ棒グラフ
height1 = np.array([ 100, 200, 300, 400, 500])
height2 = np.array([1000, 800, 600, 400, 200])
x = np.array([1,2,3,4,5])
plt.figure(figsize=(10,6))
p1 = plt.bar(x, height1, color='blue')
p2 = plt.bar(x, height2, bottom=height1, color='lightblue')

plt.legend((p1[0], p2[0]),('Class 1', 'Class 2'))

# 円グラフ
labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.figure(figsize=(15,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

# バブルチャート
N = 25
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = 10*np.pi*(15*np.random.rand(N)) **2
plt.figure(figsize=(15,6))
plt.scatter(x,y,s=area, c=colors, alpha=0.5)
plt.grid(True)
plt.show()
