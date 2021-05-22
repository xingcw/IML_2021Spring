import csv
import numpy as np 
import sklearn
import pandas as pd
from scipy import stats
from scipy.stats import uniform
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFECV
PARAM_DIST = {
    "n_estimators": stats.randint(150, 500),
    "learning_rate": stats.uniform(0.01, 0.07),
    "max_depth": [4, 5, 6, 7],
    "colsample_bytree": stats.uniform(0.5, 0.45),
    "min_child_weight": [1, 2, 3, 4, 5, 6],
    "gamma": [0.1, 0.2, 0.3]}
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
    return tf_np
def write_label(label,column,filename):
    label_pd = pd.DataFrame(data = label,columns=column)
    label_pd.to_csv(filename,index=False)
def trainning(tf_np,tl_np):
    x_train, x_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,0], test_size = 0.3,random_state=1)
    pid =list(y_test.reshape(1,-1))
    y_pred = list(y_test.reshape(1,-1))
    for i in range(1,11):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.3,random_state=1)
        '''
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test)
        clf = RandomForestClassifier(n_estimators = 500,max_depth=4,class_weight = 'balanced,me)
        clf.fit(X_new, y_train)
        y_pred.append(clf.predict(X_test))
        pid.append(y_test)
        '''
        y_pred.append(y_test)
        pid.append(y_test)
    for i in range(11,12):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:], tl_np[:,i], test_size = 0.3,random_state=1)
        '''
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test)
        #clf = RandomForestClassifier(n_estimators = 750,max_depth=4,class_weight = 'balanced')
        #clf.fit(X_new, y_train)
        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel).fit(X_new, y_train)
        '''

        clf.fit(x_train, y_train)
        y_pred.append(clf.predict(x_test))
        pid.append(y_test)
        #print (np.mean((y_pred-y_test)**2))
        
        
        #y_pred.append(y_test)
        #pid.append(y_test)
    for i in range(12,16):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.3,random_state=1)
        '''
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test)
        estimator = svm.SVR()
        estimator = estimator.fit(X_new, y_train)
        y_pred.append(y_test)
        '''
        #y_pred.append(estimator.predict(X_test))
        y_pred.append(y_test)
        pid.append(y_test)
        #print (np.mean((y_pred-y_test)**2))
    y_pred = np.array(y_pred)
    pid = np.array(pid)
    return y_pred.T,pid.T
def task2(tf_np,tl_np):
    #0.66
    for i in range(11,12):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:], tl_np[:,i], test_size = 0.3,random_state=1)
        logistic = RandomForestClassifier(n_estimators=415,class_weight='balanced',max_features='log2')
        distributions = dict(criterion=["gini","entropy"],max_depth=range(2,10),min_samples_split=range(2,5))
        clf = RandomizedSearchCV(logistic, distributions, random_state=0,scoring = 'roc_auc')
        search = clf.fit(x_train, y_train)
        y_pred = search.predict(x_test)
        print (roc_auc_score(y_test,y_pred),search.best_params_)
def task3(tf_np,tl_np):
    for i in range(12,16):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:], tl_np[:,i], test_size = 0.3,random_state=1)
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test)
        estimator = svm.SVR()
        estimator = estimator.fit(X_new, y_train)
        res = estimator.predict(X_test)
        print (0.5 + 0.5 * np.maximum(0, metrics.r2_score(y_test, res)))
        
tf_np = read_feature('train_new_feature.csv')
tl = pd.read_csv('train_labels.csv',sep=',',header=0)
column = tl.columns
tl = tl.astype(float)
tl = tl.sort_values(['pid'])
tl_np = np.asarray(tl)
task3(tf_np,tl_np)
#trainning(tf_np,tl_np)
#write_label(y_pred,column,'1.csv')
#write_label(y_test,column,'2.csv')


        