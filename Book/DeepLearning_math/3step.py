"""
Logistic Regression
"""
import numpy as np

data = [[0,0],[2,0],[3,0], [5,0],[6,1],[7,1],[8,1],[9,1],[11,1]]

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dEa(a,b):
    sum = 0
    for i in data:
        sum += i[0]*(sigmoid(a*i[0]+b)-i[1])
    return sum

def dEb(a,b):
    sum = 0
    for i in data:
        sum += sigmoid(a*i[0]+b)-i[1]
    return sum

eta = 0.5 # Learning rate
a, b = 2, 1 # Start point

for i in range(0,200):
    a += - eta * dEa(a,b)
    b += - eta * dEb(a,b)
    print([b/a])
