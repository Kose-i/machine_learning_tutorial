import numpy as np

T = np.array([[0.10, 0.70, 0.20], [0.75, 0.15, 0.10], [0.60, 0.35, 0.05]])
X = np.array([1.0, 0.0, 0.0])
x = X.dot(T).dot(T).dot(T)
print(x)

import matplotlib.pyplot as plt

data = np.loadtxt('datasets/data_1D.txt', delimiter=',')
X = np.column_stack([data[:,2]])

plt.plot(np.arange(X.shape[0]), X[:,0], c='black')
plt.title('Training data')
plt.show()

from hmmlearn.hmm import GaussianHMM

num_components = 10
hmm = GaussianHMM(n_components=num_components, covariance_type='diag', n_iter=1000)

print('Training the Hidden Markov Model...')
hmm.fit(X)

print('Means and variances:')
for i in range(hmm.n_components):
    print('\nHidden state', i+1)
    print('Mean =', round(hmm.means_[i][0],2))
    print('Variance =', round(np.diag(hmm.covars_[i])[0], 2))

num_samples = 1200
generated_data, _ = hmm.sample(num_samples)
plt.plot(np.arange(num_samples), generated_data[:,0], c='black')
plt.title('Generated data')
plt.show()
