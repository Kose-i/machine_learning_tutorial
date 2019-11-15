import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

text = np.loadtxt('datasets/data_perceptron.txt')

data   = text[:, :2]
labels = text[:,2]

plt.figure()
plt.scatter(data[labels==0, 0], data[labels==0, 1], marker='o')
plt.scatter(data[labels==1, 0], data[labels==1, 1], marker='x')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
plt.show()

dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

num_output = 1
perceptron = nl.net.newp([dim1, dim2], num_output)
error_progress = perceptron.train(data, labels.reshape(-1,1), epochs=100, show=20, lr=0.03)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

xy = np.random.rand(100, 2)
out = perceptron.sim(xy).ravel()

plt.figure()
plt.scatter(xy[out==0,0], xy[out==0,1], marker='o')
plt.scatter(xy[out==1,0], xy[out==1,1], marker='x')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

text = np.loadtxt('datasets/data_simple_nn.txt')
data   = text[:,0:2]
labels = text[:,2:]

def plot4(data, labels):
    plt.figure()
    ind = labels[:,0]*2 + labels[:,1]
    plots = []
    for i,m in enumerate(('o', '.', '+', 'x')):
        p = plt.scatter(data[ind==i,0], data[ind==i,1], marker=m, c='black')
        plots.append(p)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Input data')
    plt.legend(plots, ['0 0', ' 0 1', '1 0', '1 1'])
    plt.show()
plot4(data, labels)

dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
num_output = 2
nn = nl.net.newp([dim1, dim2], num_output)
error_progress = nn.train(data, labels, epochs=100, show=20, lr=0.03)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

print('\nTest results:')
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])

xy = np.array(data_test)
out = np.where(nn.sim(xy) < 0.5, 0, 1)
plot4(xy, out)

x = np.random.rand(100) * (dim1_max - dim1_min) + dim1_min
y = np.random.rand(100) * (dim2_max - dim2_min) + dim2_min
xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
out = np.where(nn.sim(xy) < 0.5, 0, 1)
plot4(xy, out)

nn = nl.net.newff([dim1, dim2], [8, num_output])
nn.trainf = nl.train.train_gd

min_val = -15
max_val =  15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3*np.square(x) + 5
y /= np.linalg.norm(y)

data   = x.reshape(-1, 1)
labels = y.reshape(-1,1)

plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()

output = nn.sim(data)

plt.figure()
plt.scatter(data, labels, marker='.')
plt.scatter(data, output)
plt.title('Actual vs predicted')
plt.show()

text = np.loadtxt('datasets/data_vector_quantization.txt')
data   = text[:, 0:2]
labels = text[:, 2:]

num_input_neurons  = 10
num_output_neurons = 4

weights = [1/num_input_neurons] * num_output_neurons
nn = nl.net.newlvq(nl.tool.minmax(data), num_input_neurons, weights)
nn.train(data, labels, epochs=500, goal=-1)

xx, yy = np.meshgrid(np.arange(0, 10, 0.2), np.arange(0, 10, 0.2))
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
grid_xy = np.hstack([xx, yy])
grid_eval = nn.sim(grid_xy)

grid_1 = grid_xy[grid_eval[:, 0] == 1]
grid_2 = grid_xy[grid_eval[:, 1] == 1]
grid_3 = grid_xy[grid_eval[:, 2] == 1]
grid_4 = grid_xy[grid_eval[:, 3] == 1]

plt.plot(grid_1[:,0], grid_1[:,1], 'm.',
         grid_2[:,0], grid_2[:,1], 'bx',
         grid_3[:,0], grid_3[:,1], 'c^',
         grid_4[:,0], grid_4[:,1], 'y+',
        )
class_1 = data[labels[:,0] == 1]
class_2 = data[labels[:,1] == 1]
class_3 = data[labels[:,2] == 1]
class_4 = data[labels[:,3] == 1]
plt.plot(class_1[:,0], class_1[:,1], 'ko',
         class_2[:,0], class_2[:,1], 'ko',
         class_3[:,0], class_3[:,1], 'ko',
         class_4[:,0], class_4[:,1], 'ko',
        )

plt.axis([0, 10, 0, 10])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Vector quantization')
plt.show()
