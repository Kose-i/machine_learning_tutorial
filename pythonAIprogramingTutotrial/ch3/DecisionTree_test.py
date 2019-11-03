import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))
    plt.show()

input_file = 'datasets/data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:,:-1], data[:,-1]

class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:,0], class_1[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)

params = {'random_state':0, 'max_depth':4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

class_names = ['Class-0', 'Class-1']
print("Classifier performance on training dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

input_file = 'datasets/data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:,:-1], data[:,-1]
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:,0], class_1[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:,0], class_2[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='^')
plt.title('Input data')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
params = {'n_estimators':100, 'max_depth':4, 'random_state':0}
classifier = RandomForestClassifier(**params)
#classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

class_names = ['Class-0','Class-1','Class-2']
print("Classifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("Classifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))

test_datapoints = np.array([[5,5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
print("Confidence measure:")
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nDatapoint:', datapoint)
    print('Probablities:', probabilities)
    print('Predicted class:', predicted_class)
visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 'Test datapoints')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtractTreesClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report

input_file = 'datasets/data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X ,y = data[:,:-1], data[:, -1]

class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:,0], class_1[:,1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
xparams = {'n_estimators':100, 'max_depth':4, 'random_state':0, 'class_weight':'balanced'}
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

class_names = ['Class-0', 'Class-1']
print("Classifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_name))
