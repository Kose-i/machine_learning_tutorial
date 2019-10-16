import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

np.random.seed(0)

# ベルヌーイ分布
prob_be_data = np.array([])
coin_data = np.array([0,0,0,0,0,1,1,1])
# unique で一意な値を抽出(ここの場合は,0,1)
for i in np.unique(coin_data):
    p = len(coin_data[coin_data==i]) / len(coin_data)
    print(i, 'が出る確率:',p)
    prob_be_data = np.append(prob_be_data, p)

# 二項分布
x = np.random.binomial(30, 0.5, 1000)
plt.hist(x)
plt.grid(True)
plt.show()

# ポアソン分布
x = np.random.poisson(7, 1000)
plt.hist(x)
plt.grid(True)
plt.show()

# 正規分布
x = np.random.normal(5, 10, 10000)
plt.hist(x)
plt.grid(True)
plt.show()

# 対数正規分布
x = np.random.lognormal(30, 0.4, 1000)
plt.hist(x)
plt.grid(True)
plt.show()

# カーネル密度関数
import requests
import zipfile
from io import StringIO
import io
zip_file_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
r = requests.get(zip_file_url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

student_data_math = pd.read_csv('student-mat.csv', sep=';')
student_data_math.absences.plot(kind='kde', style='k--')

student_data_math.absences.hist(density=True)
plt.grid(True)
plt.show()
