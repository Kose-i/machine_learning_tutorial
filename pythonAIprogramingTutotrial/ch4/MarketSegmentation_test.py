import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

input_file = 'datasets/sales.csv'
file_reader = csv.reader(open(input_file, 'r'), delimiter=',')

X = []
for count, row in enumerate(file_reader):
    if not count:
        names = row[1:]
        continue
    X.append([float(x) for x in row[1:]])
X = np.array(X)
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X))

meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_model.fit(X)

labels = meanshift_model.labels_
cluster_centers = meanshift_model.cluster_centers_
num_clusters = len(np.unique(labels))

print("\nNumber of clusters in input data =", num_clusters)
print("\nCenters of clusters:")
print('\t'.join(name[:7] for name in names))
for cluster_center in cluster_centers:
    print('\t'.join([str(int(x)) for x in cluster_center]))

plt.figure()
x = 1
y = 2
plt.scatter(cluster_centers[:,x], cluster_centers[:,y], s=120, edgecolor='black', facecolors='none')
plt.title('Centers of 2D clusters')
plt.xlabel(names[x])
plt.ylabel(names[y])
plt.show()
