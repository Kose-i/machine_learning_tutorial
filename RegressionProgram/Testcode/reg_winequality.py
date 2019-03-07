#!/usr/bin/env python3

import numpy as np
import csv

import linearreg

Xy = []
print("Load file")
with open("../../dataset/winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)
print("Finish load file")
Xy = np.array(Xy[1:], dtype=np.float64)

np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_X = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]

model = linearreg.LinearRegression()
model.fit(train_X, train_y)

y = model.predict(test_X)

print("first 5 predict and answer")
for i in range(5):
    print("answer:{:1.0f} predict:{:5.3f}".format(test_y[i], y[i]))
print()
print("RMSE:", np.sqrt( ((test_y - y)**2).mean() ))
