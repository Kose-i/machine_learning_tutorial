import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

hier_df = DataFrame(
    np.arange(9).reshape((3,3)),
    index = [
      ['a','a','b'],
      [1,2,2]
    ],
    columns = [
      ['Osaka','Tokyo','Osaka'],
      ['Blue','Red','Red']
    ])
print(hier_df)
hier_df.index.names   = ['key1','key2']
hier_df.columns.names = ['city','color']
print(hier_df)
print(hier_df['Osaka'])
print(hier_df.sum(level='key2', axis=0))
print(hier_df.sum(level='color', axis=1))
hier_df.drop(['b'])
print(hier_df)


