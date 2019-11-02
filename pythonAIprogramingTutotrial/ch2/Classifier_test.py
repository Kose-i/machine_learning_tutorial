#  Logistic Regression

import numpy as np
from sklearn import linear_model

X = np.array([[3.1, 7.2], [4, 6.7], [2.8, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

classifier = linear_model.LogisticRegression(solver='liblinear', C=1, multi_class='auto')
classifier.fit(X, y)

import matplotlib.pyplot as plt
def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    plt.scatter(X[:,0], X[:,1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))

    plt.show()

visualize_classifier(classifier, X, y)
classifier = linear_model.LogisticRegression(solver='liblinear', C=100, multi_class='auto')

# Naive Bayes

import numpy as np
from sklearn.naive_bayes import GaussianNB

input_file = 'datasets/data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:,:-1], data[:, -1]

classifier = GaussianNB()
classifier.fit(X, y)

y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier, X, y)

# cross validation

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)

classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)

y_test_pred = classifier_new.predict(X_test)

accuracy = 100.0*(y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier_new, X_test, y_test)

num_folds = 3
accuracy_values  = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " +str(round(100*accuracy_values.mean(), 2)) + "%")
precision_values = model_selection.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: "+str(round(100*precision_values.mean(), 2))+"%")
recall_values    = model_selection.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: "   +str(round(100*recall_values.mean(), 2))+"%")
f1_values        = model_selection.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: "       +str(round(100*f1_values.mean(), 2))+"%")

# confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(true_labels, pred_labels)
print(confusion_mat)
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.xlabel('True label')
plt.ylabel('Predicted labels')
plt.show()
targets = ['Class-0','Class-1','Class-2','Class-3','Class-4']
print(classification_report(true_labels, pred_labels, target_names=targets))
