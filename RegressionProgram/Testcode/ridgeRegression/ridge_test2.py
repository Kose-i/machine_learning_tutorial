#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import linearreg

import ridge

x = np.arange(12)
y = 1 + 2*x
y[2] = 20
y[4] = 0

xmin, xmax = 0, 12
ymin, ymax = -1, 25
fig, axes = plt.subplots(nrows=2, ncols=5)
for i in range(5):
    xx = x[:2+i*2]
    yy = y[:2+i*2]

    axes[0,i].set_xlim([xmin, xmax])
    axes[0,i].set_ylim([ymin, ymax])
    axes[0, i].scatter(xx, yy, color="k")
    model = linearreg.LinearRegression()
    model.fit(xx, yy)
    xs = [xmin, xmax]
    ys = [model.w_[0] + model.w_[1]*xmin, model.w_[0] + model.w_[1]*xmax]
    axes[0, i].plot(xs, ys, color="k")

    axes[1,i].set_xlim([xmin, xmax])
    axes[1,i].set_ylim([ymin, ymax])
    axes[1, i].scatter(xx, yy, color="k")
    model = ridge.RidgeRegression(10.)
    model.fit(xx, yy)
    xs = [xmin, xmax]
    ys = [model.w_[0] + model.w_[1]*xmin, model.w_[0] + model.w_[1]*xmax]
    axes[1, i].plot(xs, ys, color="k")

plt.show()
