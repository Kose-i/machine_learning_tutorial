import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set()

import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.5, random_state=66)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)

print('Cross validation scores: {}'.format(scores))
print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))

# AUC ROC
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)
results = pd.DataFrame(model.predict_proba(X_test), columns=cancer.target_names)
results.head()

rates = {}
for threshold in np.linspace(0.01, 0.99, num=50):
    labels = results['benign'].map(lambda x: 1 if x>threshold else 0)
    m = confusion_matrix(y_test, labels)
    rates[threshold] = {'false positive rate':m[0,1]/m[0,:].sum(), 'true positive rate':m[1,1]/m[1,:].sum()}
pd.DataFrame(rates).T.plot.scatter('false positive rate','true positive rate')

from sklearn import svm
from sklearn.metrics import ros_curve, auc
model = svm.SVC(kernel='linear', probability=True, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='red', label='ROC curve (area=%.3f)'%auc)
plt.plot([0,1],[0,1],color='black', linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.xlabel('True  positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="best")
plt.show()
