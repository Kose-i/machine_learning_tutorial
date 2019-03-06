#!/usr/bin/env python3

import cvxopt
import numpy as np
"""
Minimize f(x, y) = x^2 + x*y + y^2 + 2*x + 4*y
              = 1/2*(x y)((2 1)(1 2))((x y)T) + (2 4)((x y)T)
"""

P = cvxopt.matrix(np.array([[2,1],[1,2]], dtype=np.float64))

q = cvxopt.matrix(np.array([2,4], dtype=np.float64))

sol = cvxopt.solvers.qp(P, q)

print("sol[\"x\"]=", np.array(sol["x"]))
print("sol[\"primal objectice\"]=", np.array(sol["primal objective"]))

"""
Minimize f(x, y) = x^2 + x*y + y^2 + 2*x + 4*y

Subject to g(x, y) = x+y = 0
                   = (1 1)((x y)T)
"""

P = cvxopt.matrix(np.array([[2,1],[1,2]], dtype=np.float64))

q = cvxopt.matrix(np.array([2,4], dtype=np.float64))

A = cvxopt.matrix(np.array([[1,1]], dtype=np.float64))

b = cvxopt.matrix(np.array([0], dtype=np.float64))

sol = cvxopt.solvers.qp(P, q, A=A, b=b)

print("sol[\"x\"]=", np.array(sol["x"]))
print("sol[\"primal objectice\"]=", np.array(sol["primal objective"]))

"""
Minimize f(x, y) = x^2 + x*y + y^2 + 2*x + 4*y

Subject to g(x, y) = 2*x + 3*y <= 3
                   = (2 3)((x y)T) <= (3)
"""

P = cvxopt.matrix(np.array([[2,1],[1,2]], dtype=np.float64))

q = cvxopt.matrix(np.array([2,4], dtype=np.float64))

G = cvxopt.matrix(np.array([[2,3]], dtype=np.float64))

h = cvxopt.matrix(np.array([3], dtype=np.float64))

sol = cvxopt.solvers.qp(P, q, G=G, h=h)

print("sol[\"x\"]=", np.array(sol["x"]))
print("sol[\"primal objectice\"]=", np.array(sol["primal objective"]))
