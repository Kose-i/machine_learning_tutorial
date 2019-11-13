import numpy as np
import pandas as pd

def read_data(input_file, i):
    def to_data(x, y):
        return str(int(x)) + '-' + str(int(y))
    input_data = np.loadtxt(input_file, delimiter=',')
    start = to_data(input_data[0,0], input_data[0,1])
    if input_data[-1, 1]==12:
        year = input_data[-1,0] + 1
        month = 1
    else:
        year = input_data[-1,0]
        month = input_data[-1,1]+1
    end = to_data(year, month)
    data_indices = pd.date_range(start, end, freq='M')
    output = pd.Series(input_data[:,i], index=data_indices)
    return output

import matplotlib.pyplot as plt
input_file = 'datasets/data_2D.txt'
indices = [2,3]

for index in indices:
    timeseries = read_data(input_file, index)

    plt.figure()
    timeseries.plot()
    plt.title('Dimension ' + str(index - 1))
plt.show()

index = 2
data = read_data('datasets/data_2D.txt', index)

start = '2003'
end   = '2011'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)
plt.show()

start = '1998-2'
end   = '2006-7'
plt.figure()
data[start:end].plot()
plt.title('Input dadta from ' + start + ' to ' + end)
plt.show()

d = pd.Series([0,1,2])
print(d[0:2])

d = pd.Series([0,1,2], index = ['foo', 'bar', 'zot'])
print(d['foo':'zot'])

x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

data = pd.DataFrame({'dim1':x1, 'dim2':x2})

start = '1968'
end   = '1975'
data[start:end].plot(style=['-', '--'])
plt.title('Data overlapped on top of each other')
plt.show()

data[(data['dim1'] < 45) & (data['dim2'] > 30)][start:end].plot(style=['-', '--'])
plt.title('dim1 < 45 and dim2 > 30')
plt.show()

sum_ = data[start:end]['dim1'] + data[start:end]['dim2']
sum_.plot()
plt.title('Summation (dim1+dim2)')
plt.show()

print('Maximum values for each dimension:\n', data.max())
print('Minimum values for each dimension:\n', data.min())
print('Overall mean:\n', data.mean())
print('\nRow-wise mean:\n', data.mean(axis=1)[:12])

start = '1968'
end   = '1975'
data[start:end].plot(style=['-', '--'])
plt.title('Original')
plt.show()
data[start:end].rolling(window=24).mean().plot(style=['-','--'])
plt.title('Rolling mean')
plt.show()

print('Correlation coefficients:\n', data.corr())
data['dim1'].rolling(window=60).corr(other=data['dim2']).plot()
plt.title('Rolling correlation')
