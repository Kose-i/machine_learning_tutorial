import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(1000)
y = map(lambda t: 0 if t < 0.5 else 1, x)

from collections import Counter

c = Counter(y)
print(c)
