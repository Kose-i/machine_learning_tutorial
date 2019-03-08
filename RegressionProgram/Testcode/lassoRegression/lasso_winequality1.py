#!/usr/bin/env python3
import numpy as np
import csv

import lasso

Xy = []
with open("../../../dataset/winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)
Xy = np.array(Xy[1:], dtype=np.float64)

np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000,  -1]
test_X  = Xy[ -1000, :-1]
test_y  = Xy[ -1000,  -1]

for lambda_ in [1., 0.1, 0.01]:
    model = lasso.Lasso(lambda_)
    model.fit(train_X, train_y)
    y = model.predict(test_X)
    print("--- lambda = {}".format(lambda_))
    print("coefficients:")
    print(model.w_)
    mse = ((y - test_y)**2).mean()
    print("MSE: {:.3f}".format(mse))
