import numpy as np
import pandas as pd
def read_feature(filename):
    tf=pd.read_csv(filename, sep=',',header=0)
    tf = tf.astype(float)
    tf2 = tf.groupby(['pid'])
    tf3= []
    for p in tf2:
        pid = np.asarray(p[1].values[0,0])
        age = np.asarray(p[1].values[0,2])
        data = np.asarray(p[1].values[:,3:])
        tmp = np.concatenate((p[1].values[:,1].reshape(-1,1),data),axis=1).reshape(-1)
        tmp = np.append(age,tmp)
        tmp = np.append(pid,tmp)
        tf3.append(tmp)
    tf_np = np.asarray(tf3)
    print (np.max(tf_np[:,0]))
    return tf_np
'''   
data = pd.read_csv('test_label.csv', sep=',',header=0)
columns = data.columns[1:]
data = data.values[:,1:]
data = pd.DataFrame(data,columns=columns)
data.to_csv('test_label2.csv',index=None)
'''
f = pd.read_csv('test_label.csv', sep=',',header=0)
f=f.round(4)
columns = f.columns
f.to_csv('test_label2.csv',index=None)
#data = data.values
#data = data[1:,:]
#print (data)