import numpy as np

sample_array1 = np.array([[1,2,3],[4,5,6]])
sample_array2 = np.array([[7,8,9],[10,11,12]])

# merge

print(sample_array1)
print(sample_array2)

concate_12 = np.concatenate([sample_array1, sample_array2],axis=0)
print(concate_12)

vstack_12  = np.vstack((sample_array1, sample_array2))
print(vstack_12)

hstack_12 = np.hstack((sample_array1, sample_array2))
print(hstack_12)

# split
first, second, third=np.split(vstack_12, [1,3])
print("first:", first)   # [:1]
print("second:", second) # [1:3]
print("third:", third)   # [3:]


