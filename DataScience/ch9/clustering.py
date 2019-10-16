import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn

import sklearn

# k-means
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X, _ = make_blobs(random_state=10)
plt.scatter(X[:,0], X[:,1], color='black')
kmeans = KMeans(init='random',n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)
merge_data = pd.concat([pd.DataFrame(X[:,0]), pd.DataFrame(X[:,1]), pd.DataFrame(y_pred)], axis=1)

merge_data.columns = ['feature1','feature2','feature3']

ax = None
colors = ['blue', 'red', 'green']
for i, data in merge_data.groupby('cluster'):
    ax = data.plot.scatter(x='feature1', y='feature2',color=colors[i], label=f'cluster{i}',ax=ax)

# エルボー法
dist_list = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='random',random_state=0)
    kmeans.fit(X)
    dist_list.append(kmeans.inertia_)
plt.plot(range(1,10), dist_list, marker='+')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# 主成分分析(PCA)
from sklearn.preprocessing import StandardScaler
sample = np.random.RandomState(1)

X = np.dot(ssample.rand(2,2), sample.randn(2,200)).T

sc = StandardScaler()
X_std = sc.fit_transform(X)

print('相関係数{:.3f}'.format(sp.stats.pearsonr(X_std[:,0], X_std[:,1])[0]))
plt.scatter(X_std[:,0], X_std[:,1])
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_std)
print(pca.components_)
print('各主成分の分散:{}'.format(pca.explained_variance_))
print('各主成分の分散割合:{}'.format(pca.explained_variance_ratio_))
arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0,shrinkB=0)
def draw_vector(v0, v1):
    plt.gca().annotate('', v1, v0, arrowprops=arrowprops)
plt.scatter(X_std[:,0],X_std[:,1],alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector*3*np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_+v)
plt.axis('equal')
plt.show()
