import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sklearn

import requests, zipfile
import io

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
res = requests.get(url).content

auto = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)
auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type','aspiration','num-of-doors','body-style', 'drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
print('自動車データの形成:{}'.format(auto.shape))
print(auto.head())

print(auto.isin(['?']).sum())
auto = auto.replace('?',np.nan).dropna()

print("相関:{}".format(auto.corr()))

# SVM
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=0)

model = LinearSVC()
model.fit(X_train, y_train)
print('正解率(train):{:.3f}'.format(model.score(X_train,y_train)))
print('正解率( test):{:.3f}'.format(model.score(X_test ,y_test )))
