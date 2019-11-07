## Pipeline

from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

X, y = samples_generator.make_classification(n_samples=150, n_features=25, n_classes=3, n_informative=6, n_redundant=0, random_state=7)
k_best_selector = SelectKBest(f_regression, k=9)

classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)
processor_pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])

processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)
processor_pipeline.fit(X, y)

output = processor_pipeline.predict(X)
print("Predicted output:\n", output)

status = processor_pipeline.named_steps['selector'].get_support()

selected = [i for i,x in enumerate(status) if x]
print("\nIndices of selected features:", ','.join([str(x) for x in selected]))

## Nearest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

plt.figure()
plt.title('Input data')
plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='black')
plt.show()

k = 5
test_datapoint = [4.3, 2.7]

knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
distances, indices = knn_model.kneighbors([test_datapoint])

print("K Nearest Neighbors:")
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " ==>", X[index])
plt.figure()
plt.title('Nearest neighbors')
plt.scatter(X[:,0], X[:,1], marker='o', s=75, color='k')
plt.scatter(X[indices][0][:,0], X[indices][0][:,1], marker='o', s=250, color='k', facecolors='none')
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', s=75, color='k')
plt.show()

## k-NN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets

input_file = 'datasets/data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(np.int)

plt.figure()
plt.title('Input data')
marker_shapes = 'v^os'

for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i]], s=75, edgecolors='black', facecolors='none')
plt.show()

num_neighbors = 12
classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
classifier.fit(X, y)
step_size = 0.01
x_min, x_max = X[:, 0].min() -1, X[:,0].max() +1
y_min, y_max = X[:, 1].min() -1, X[:,1].max() +1

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

plt.figure()
plt.pcolormesh(x_values, y_values, output, cmap=cm.Blues)

for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i]], s=50, edgecolors='black', facecolors='none')
plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())
plt.title('K Nearest Neighbors classifier model boundaries')
plt.show()

test_datapoint = [5.1, 3.6]
plt.figure()
plt.title('Test datapoint')
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i]], s=75, edgecolors='black', facecolors='none')
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=6, s=200, facecolors='black')
plt.show()

_, indices = classifier.kneighbors([test_datapoint])
indices = indices.astype(np.int)[0]

plt.figure()
plt.title('K Nearest Neighbors')

for i in indices:
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i]], linewidth=3, s=100, facecolors='black')
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=6, s=200, facecolors='black')
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i]], s=75, edgecolors='black', facecolors='none')
plt.show()

print('K-Nearest Neighbours:')
for i in indices:
    print('({}, {}) -> {}'.format(X[i, 0], X[i, 1], y[i]))
print("Predicted output:", classifier.predict([test_datapoint])[0])
