#datasets:https://archive.ics.uci.edu/ml/datasets/Census+Income

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import model_selection

input_file = 'datasets/income_data.txt'
Xy = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        for i,_ in enumerate(data):
            data[i] = data[i].lstrip(' ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            Xy.append(data)
            count_class1 += 1
        if data[-1] == '>50K'  and count_class2 < max_datapoints:
            Xy.append(data)
            count_class2 += 1
Xy = np.array(Xy)
label_encoder = []
Xy_encoded = np.empty(Xy.shape)
for i, item in enumerate(Xy[0]):
    if item.isdigit():
        Xy_encoded[:,i] = Xy[:,i]
    else:
        encoder = preprocessing.LabelEncoder()
        Xy_encoded[:,i] = encoder.fit_transform(Xy[:,i])
        label_encoder.append(encoder)
X = Xy_encoded[:, :-1].astype(int)
y = Xy_encoded[:,  -1].astype(int)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)
classifier = LinearSVC(random_state=0)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
f1 = model_selection.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score:"+str(round(100*f1.mean(), 2)) + "%")
input_data = np.array([
    ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'],
    ['55', 'Private', '287927', 'Doctorate', '16', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Female', '15000', '0', '40', 'United-States']])
input_data_encoded = np.zeros(input_data.shape)
c = 0
for i,item in enumerate(input_data[0]):
    if item.isdigit():
        input_data_encoded[:,i] = input_data[:,i]
    else:
        input_data_encoded[:,i] = label_encoder[c].transform(input_data[:,i])
        c += 1
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class))
