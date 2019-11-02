import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

RANGE = 5
X = np.random.rand(100)*RANGE
y = np.sin(X) + np.random.rand(100)*0.3

X = X.reshape(-1, 1)
regressor = linear_model.LinearRegression()
regressor.fit(X, y)

x_test = np.linspace(0, RANGE, 100).reshape(-1,1)
y_pred = regressor.predict(x_test)

plt.scatter(X, y, color='green')
plt.plot(x_test, y_pred, color='black')
plt.show()

y_pred = regressor.predict(X)
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))
print("Mean squared  error =", round(sm.mean_squared_error(y, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_pred), 2))

from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=10)
x_transformed = polynomial.fit_transform(X)
poly_linear_regressor = linear_model.LinearRegression()
poly_linear_regressor.fit(x_transformed, y)

x_test_transformed = polynomial.transform(x_test)
y_pred = poly_linear_regressor.predict(x_test_transformed)
plt.scatter(X, y, color='green')
plt.plot(x_test, y_pred, color='black')
plt.show()

y_pred = poly_linear_regressor.predict(x_transformed)
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))
print("Mean squared  error =", round(sm.mean_squared_error(y, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y, y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_pred), 2))

import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

data = datasets.load_boston()
X, y = shuffle(data.data, data.target, random_state=7)
num_training = int(0.8*len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print("Predicted price", sv_regressor.predict([test_data])[0])
