import numpy as np
import scipy as sp
import scipy.linalg as linalg

A = np.array([[1,2,3,4,5],[6,7,8,9,10]])

# SVD分解
U, s, Vs = sp.linalg.svd(A)
m,n = A.shape

S = sp.linalg.diagsvd(s,m,n)
print('U.S.V*=\n',U@S@Vs)

# LU分解
A = np.identity(5)
A[0,:] = 1
A[:,0] = 1
A[0,0] = 5
b = np.ones(5)

# 正方行列をLU分解する
(LU, piv) = sp.linalg.lu_factor(A)
L = np.identity(5) + np.tril(LU, -1)
U = np.triu(LU)
P = np.identity(5)[piv]
# 解を求める
x = sp.linalg.lu_solve((LU,piv),b)
print(x)

# コレスキー分解
A = np.array([[7,-1,0,1],
              [-1,9,-2,2],
              [0,-2,8,-3],
              [1,2,-3,10]
              ])
b = np.array([5,20,0,20])
L = sp.linalg.cholesky(A)
t = sp.linalg.solve(L.T.conj(), b)
x = sp.linalg.solve(L, t)
print(x)
# 確認
print(np.dot(A,x))

# NMFを使う
from sklearn.decomposition import NMF
# 分解対象行列
X = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
model = NMF(n_components=2, init='random',random_state=0)
W = model.fit_transform(X)
H = model.components_
print(W)
print(H)
print(np.dot(W,H)) # W@Hでもよい
