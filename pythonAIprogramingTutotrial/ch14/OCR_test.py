import cv2
import numpy as np
import matplotlib.pyplot as plt

input_file = 'datasets/letter.data'

W, H = 8, 16
W1, H1 = W+1, H+1
N_COLS, N_ROWS = 9, 5
simg = np.zeros((H1*N_ROWS, W1*N_COLS), dtype=np.uint8)

col, row = 0, 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        c = list_vals[1]
        print(c, end='')
        data = np.array([255*int(x) for x in list_vals[6:-1]])
        img = np.reshape(data, (H, W)).astype(np.uint8)
        simg[row*H1:row*H1+H, col*W1:col*W1+W] = img
        col += 1
        if col == N_COLS:
            col = 0
            row += 1
            print('')
            if row == N_ROWS:
                break
simg = cv2.cvtColor(simg, cv2.COLORGRAY2RGB)

plt.figure()
plt.imshow(simg)
plt.show()

import neurolab as nl

num_datapoints = 50
num_train = int(0.9*num_datapoints)
num_test  = num_datapoints - num_train

orig_labels = 'omandig'
num_orig_labels = len(orig_labels)

data = []
labels = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels:
            continue
        label = np.zeros((num_orig_labels, 1))
        label[orig_labels.inex(list_vals[1])] = 1
        labels.append(label)
        cur_char = np.array([float(x) for x in list_vals[0:-1]])
        data.append(cur_char)
        if len(data) >= num_datapoints:
            break
labels = np.array(labels).reshape(-1, num_orig_labels)
num_dims = 8*16
data = np.array(data).reshape(-1, num_dims)

nn = nl.net.newff([[0, 1]]*num_dims, [128, 16, num_orig_labels])
nn.trainf = nl.train.train_gd

error_progress = nn.train(data[:num_train, :], labels[:num_train,:], epochs=10000, show=100, goal=0.01)

print('Testing on unknown data')
predicted = nn.sim(data[num_train:,:])
for i in range(num_test):
    print('\nOriginal:', orig_labels[np.argmax(labels[num_train+1])])
    print('Predicted:', orig_labels[np.argmax(predicted[i])])
