"""
forward Convolution layer by scipy
"""
from scipy import signal

# Input
x = [[0,1,2,1,0,0],
     [0,0,1,2,1,0],
     [0,0,0,1,2,1],
     [0,0,0,1,3,2],
     [1,1,1,3,2,0],
     [2,2,3,2,1,0]]
# Filter
f = [[0,1,1],[0,1,1],[0,1,1]]

result = signal.correlate(x,f,'valid')
print(result)
