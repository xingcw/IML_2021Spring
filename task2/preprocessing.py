import csv
import numpy as np 
import sklearn
import pandas as pd
def fillNAN():
    tf=pd.read_csv('test_features.csv', sep=',',header=0)
    #tl = pd.read_csv('train_labels.csv',sep=',',header=0)
    tf = tf.astype(float)
    tf = tf.sort_values(['pid','Time'])
    values = np.array(tf.values)
    mean = np.nanmean(values,axis=0)
    dic = dict(zip(tf.columns,mean))
    tf1 = tf.groupby(['pid'])
    tf3= []
    for p in tf1:
        p[1].fillna(method ='ffill',inplace=True)
        p[1].fillna(method = 'bfill',inplace=True)
        p[1].fillna(value=dic,inplace=True)
        feature = np.asarray(p[1].values)
        tf3.append(feature)
    tf3 = np.array(tf3)
    tf3 = np.reshape(tf3,(-1,37))
    tf3 = pd.DataFrame(tf3,columns=tf.columns)
    #tf3.fillna(value=dic,inplace=True)
    tf3.to_csv('4.csv')


fillNAN()