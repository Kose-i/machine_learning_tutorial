# 1変数回帰

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

input_file = 'datasets/data_singlevar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
regressor = linear_model.LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
plt.scatter(X, y, color='green')
plt.plot(X, y_pred, color='black')
plt.show()
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))
print("Mean squared  error =", round(sm.mean_squared_error(y, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_pred), 2))

#import pickle
#output_model_file = 'model.pkl'
#with open(output_model_file, 'wb') as f:
#    pickle.dump(regressor, f)
#with open(output_model_file, 'rb') as f:
#    regressor_model = pickle.load(f)
#y_pred_new = regressor_model.predict(y)
#print("\nNew mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))

# 多変数回帰

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm

input_file = 'datasets/data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X,y = data[:,:-1], data[:,-1]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X, y)

y_pred = linear_regressor.predict(X)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))
print("Mean squared  error =", round(sm.mean_squared_error(y, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_pred), 2))

print("Linear regression:\n", linear_regressor.predict(X[0:5]))
print("True output:\n", y[0:5])
