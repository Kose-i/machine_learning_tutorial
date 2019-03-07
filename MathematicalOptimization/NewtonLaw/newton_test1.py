#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import newton

"""
f1(x,y) = x^3 - 2y = 0
f2(x,y) = x^2 + y^2 - 1 = 0

solve x,y
"""

def f1(x, y):
    return x**3 - 2*y

def df1(x, y):
    return np.array([3*x**2,-2])

def f2(x, y):
    return x**2 + y**2 - 1

def df2(x, y):
    return np.array([2*x, 2*y])

def f(xx):
    x = xx[0]
    y = xx[1]
    return np.array([f1(x,y), f2(x,y)])

def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([df1(x,y), df2(x,y)])

xmin, xmax = -3, 3
ymin, ymax = -3, 3
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
x = np.linspace(xmin, xmax, 200)
y = np.linspace(ymin, ymax, 200)
xmesh, ymesh = np.meshgrid(x, y)
z1 = f1(xmesh, ymesh)
z2 = f2(xmesh, ymesh)
plt.contour(xmesh, ymesh, z1, colors="r", levels=[0])
plt.contour(xmesh, ymesh, z2, colors="k", levels=[0])
solver = newton.Newton(f, df)

initials = [np.array([1,1]), np.array([-1,-1]), np.array([1,-1])]
markers = ["+", "*", "x"]

for x0, m in zip(initials, markers):
    sol = solver.solve(x0)
    plt.scatter(solver.path_[:,0],solver.path_[:,1], color="k",marker=m)
    print(sol)

plt.show()