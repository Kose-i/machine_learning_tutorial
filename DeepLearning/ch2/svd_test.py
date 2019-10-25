import numpy as np

A = np.array([[1,2],[3,4]])
print("A:",A)
#a_eig = np.linalg.eig(A)
#print("Aの固有値:\n", a_eig[0])
#print("Aの固有ベクトル:\n", a_eig[1])
#a_aT_eig = np.linalg.eig(np.dot(A, A.T))
#print("A*A.Tの固有値:\n", a_aT_eig[0])
#print("U=A*A.Tの固有ベクトル:\n", a_aT_eig[1])
#aT_a_eig = np.linalg.eig(np.dot(A.T, A))
#print("A.T*Aの固有値:\n", aT_a_eig[0])
#print("V=A.T*Aの固有ベクトル:\n", aT_a_eig[1])

#[1 2][x1] = [3]
#[3 4][x2] = [7]

#y = np.array([3,7]).T

U, d, V = np.linalg.svd(A, full_matrices=True)
D = np.diag(d)

print("U:\n",U)
print("D:\n",D)
print("V.T:\n", V.T)

print("UDV:\n", np.dot(np.dot(U, D),V))

pseudoinverse_D = np.array([[1/f if f!=0 else 0 for f in t] for t in D]).T
print("Dinv:\n", pseudoinverse_D)
pseudoinverse_A = np.dot(np.dot(V,pseudoinverse_D), U.T)
print("pseudoinverse_A:\n", pseudoinverse_A)
print("inverse_A:\n", np.linalg.inv(A))
#print("y=", np.dot(A,x))
