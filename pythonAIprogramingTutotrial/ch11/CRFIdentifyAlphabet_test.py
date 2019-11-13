import string
import numpy as np
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

class CRFModel(object):
    def __init__(self, c_val=1.0):
        self.clf = FrankWolfeSSVM(model=ChainCRF(), C=c_val, max_iter=50)
    def load_data(self):
        alphabets = load_letters()
        X = np.array(alphabets['data'])
        y = np.array(alphabets['labels'])
        folds = alphabets['folds']
        return X,y,folds
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    def evaluate(self, X_test, y_test):
        return self.clf.score(X_test, y_test)
    def classify(self, input_data):
        return self.clf.predict(input_data)[0]
    def convert_to_letters(indices):
        alphabets = np.array(list(string.ascii_lowercase))
        output = np.take(alphabets, indices)
        output = ''.join(output)
        return output

crf = CRFModel(1.0)

X,y,folds = crf.load_data()
X_train, X_test = X[folds==1], X[folds!=1]
y_train, y_test = y[folds==1], y[folds!=1]

print('Training the CRF model...')
crf.train(X_train, y_train)

score = crf.evaluate(X_test, y_test)
print('Accuracy score =', str(round(score*100, 2)) + '%')

indices = range(3000, len(y_test), 200)
for index in indices:
    print("\nOriginal =", convert_to_letters(y_test[index]))
    predicted = crf.classify([X_test[index]])
    print("Predicted =", convert_to_letters(predicted))
