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
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFECV
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
def testing(tf_np,tl_np,tef_np,params,stride=10):
    x_train = tf_np[::stride]
    x_test = tef_np
    y_pred = tef_np[:,0].reshape(1,-1)
    for i in range(1,12):
        x_train, X_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.4)
        clf = RandomForestClassifier(n_estimators=415,class_weight='balanced',criterion='entropy',max_features='log2',max_depth=params[i-1,0],min_samples_split=params[i-1,1])
        search = clf.fit(x_train, y_train)
        res = search.predict(x_test[:,1:])
        y_pred = np.concatenate((y_pred,res.reshape(1,-1)),axis=0)
        print (y_pred.shape)
    for i in range(12,16):
        x_train, X_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.4)
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test[:,1:])
        estimator = svm.SVR()
        estimator = estimator.fit(X_new, y_train)
        res = estimator.predict(X_test)
        y_pred = np.concatenate((y_pred,res.reshape(1,-1)),axis=0)
    y_pred = np.array(y_pred)
    return y_pred
def trainning(tf_np,tl_np,tef_np):
    y_pred = tef_np[:,0].reshape(1,-1)
    for i in range(1,12):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.4)
        '''
        logistic = RandomForestClassifier(n_estimators=415,class_weight='balanced',max_features='log2')
        distributions = dict(criterion=['entropy','gini'],max_depth=range(3,10),min_samples_split=range(2,5))
        clf = RandomizedSearchCV(logistic, distributions,scoring = 'roc_auc')
        '''
        clf = xgb.XGBClassifier(objective='binary:logistic',max_depth= 7,learning_rate= 0.01,num_class= 1, n_estimators=1000,
                   gamma =0.2,
                   subsample=0.8,
                   colsample_bytree= 0.8,
                   scale_pos_weight=1,
                   min_child_weight=6,
                   reg_alpha=0.01)
        search = clf.fit(x_train, y_train.astype(int))
        res = search.predict_proba(x_test)[:,1]
        print (roc_auc_score(y_test,res))#,search.best_params_)
        res = search.predict_proba(tef_np[:,1:])[:,1]
        y_pred = np.concatenate((y_pred,res.reshape(1,-1)),axis=0)
    for i in range(12,16):
        x_train, x_test, y_train, y_test = train_test_split(tf_np[:,1:], tl_np[:,i], test_size = 0.4)
        a = SelectPercentile(percentile=12.5)
        X_new = a.fit_transform(x_train, y_train)
        X_test = a.transform(x_test)
        estimator = svm.SVR()
        estimator = estimator.fit(X_new, y_train)
        res = estimator.predict(X_test)
        print (0.5 + 0.5 * np.maximum(0,r2_score(y_test,res)))
        X_test = a.transform(tef_np[:,1:])
        res = estimator.predict(X_test)
        y_pred = np.concatenate((y_pred,res.reshape(1,-1)),axis=0)
    y_pred = np.array(y_pred)
    return y_pred
tf_np = read_feature('3.csv')
tef_np = read_feature('4.csv')
tl = pd.read_csv('train_labels.csv',sep=',',header=0)
params = pd.read_csv('para.csv',sep=',',header=0)
params = np.asarray(params.values)
params = params.reshape(-1,2)
column = tl.columns
tl = tl.astype(float)
tl = tl.sort_values(['pid'])
tl_np = np.asarray(tl)
para = pd.read_csv('para.csv', sep=',',header=0)
para = para.values
a = trainning(tf_np,tl_np,tef_np)
#a = testing(tf_np,tl_np,tef_np,params)
a = a.transpose()
a = pd.DataFrame(a,columns=tl.columns)
a.to_csv('test_label.csv',index=None)