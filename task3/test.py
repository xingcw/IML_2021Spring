import numpy as np

a = np.array([1,2,3,4])
b = np.array([3,4,5,6])
c = [a,b]
print(np.concatenate(c,axis=0))