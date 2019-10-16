import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

# 線形回帰
from sklearn import linear_model
reg = linear_model.LinearRegression()

student_data_math = pd.read_csv('student-mat.csv', sep=';')
X = student_data_math.loc[:,['G1']].values
Y = student_data_math['G3'].values
reg.fit(X,Y)

print('回帰係数:{0}'.format(reg.coef_))
print('切片:{0}'.format(reg.intercept_))

plt.scatter(X,Y)
plt.xlabel('G1 grade')
plt.ylabel('G3 grade')

plt.plot(X, reg.predict(X))
plt.grid(True)

# 決定係数
print('決定係数:{0}'.format(reg.score(X,Y)))
