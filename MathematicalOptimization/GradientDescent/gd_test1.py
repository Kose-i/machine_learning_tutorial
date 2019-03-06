#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gd

"""
Minimize f(x,y) = 5*x^2 - 6*x*y + 3*y^2 + 6*x - 6*y
"""

def f(xx):
    x = xx[0]
    y = xx[1]
    return (5*x**2 - 6*x*y + 3*y**2 + 6*x - 6*y)

def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([10*x - 6*y + 6, -6*x + 6*y - 6])


xmin, xmax = -2, 2
ymin, ymax = -2, 2

algo = gd.GradientDescent(f, df)
initial = np.array([1,1])
algo.solve(np.array(initial))
print("algo.x_ =" ,algo.x_)
print("algo.opt_ =",algo.opt_)

plt.scatter(initial[0], initial[1], color="k",marker="o")
plt.plot(algo.path_[:,0], algo.path_[:,1], color="k",linewidth=1.5)
xs = np.linspace(xmin, xmax, 300)
ys = np.linspace(ymin, ymax, 300)
xmesh, ymesh = np.meshgrid(xs, ys)
xx = np.r_[xmesh.reshape(1, -1), ymesh.reshape(1, -1)]
levels = [-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, 0, 1, 2, 3, 4]
plt.contour(xs, ys, f(xx).reshape(xmesh.shape), levels=levels, colors="k", linestyles="dotted")

plt.show()
