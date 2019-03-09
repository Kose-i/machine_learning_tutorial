#!/usr/bin/env python3

import numpy as np

X      = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[5,6],[7,9]])
labels = np.array([    0,    1,    2,    0,    1,    2,    0])

# スライス
print("X[labels == 0, :]=", X[labels == 0, :])

# 重心
print("X[labels == 0, :]=", X[labels == 0, :].mean(axis=0))

## 距離の二乗
cluster_centers = np.array([[1,1],[2,2],[3,3]])#center

print(((X[:,:,np.newaxis] - cluster_centers.T[np.newaxis,:,:])**2).sum(axis=1))

"""
p = X[:,:,np.newaxis]
q = cluster_centers.T[np.newaxis,:,:]
r = (p - q)**2
s = r.sum(axis=1)
print(s)

# index=1
print(s.argmin(axis=1))
"""
