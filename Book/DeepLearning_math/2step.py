"""
Approach Minimum a little move
"""

def f(x,y):
    return x*x + y*y

def dfx(x,y):
    return 2*x

def dfy(x,y):
    return 2*y

eta = 0.1 # Learning rate
x, y = 10, 8 # Initialize start point
print(f(x,y))
# Repeat Move
for i in range(0,30):
    x += -eta*dfx(x,y)
    y += -eta*dfy(x,y)
    print(f(x,y))
