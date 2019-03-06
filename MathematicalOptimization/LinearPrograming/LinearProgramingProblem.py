#!/usr/bin/env python3

import numpy as np
from scipy import optimize

"""
Minimize (-3 -4) ((x y)T)
Subject to ((1 4)(2 3)(2 1)(-1 0)(0 -1)) ((xy)T) <= (1700 1400 1000 0 0)
"""
c = np.array([-3, -4], dtype=np.float64)
G = np.array([[1,4], [2,3],[2,1]], dtype=np.float64)
h = np.array([1700,1400,1000], dtype=np.float64)
sol = optimize.linprog(c, A_ub=G, b_ub=h, bounds=(0, None))

print("sol.x = ", sol.x)
print("sol.fun = ", sol.fun)
