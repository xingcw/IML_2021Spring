#test
import numpy as np
array = [[2,np.nan,3,4,np.nan],[2,np.nan,np.nan,1,np.nan],[2,1,np.nan,np.nan,np.nan],[2,np.nan,np.nan,np.nan,np.nan]]
array = np.vstack((np.array([0,0,0,0,0]), np.array(array)))
print(array)
last_nonnan_index_list = []
for column in array.T:
    last_nonnan_index = np.where(~np.isnan(column))
    print(last_nonnan_index[-1][-1])
    last_nonnan_index_list.append(last_nonnan_index[-1])

print(last_nonnan_index_list)
print(np.zeros((3,2)))