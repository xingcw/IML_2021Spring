from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('train.csv', sep=',',header=0)
X = data['Sequence']
y = np.array(data['Active'].astype(float))
v = CountVectorizer(analyzer='char')
x = v.fit_transform(X)
feature = v.get_feature_names()
dictionary = dict(zip(feature,np.arange(len(feature))))
matrix = np.zeros((len(X),4*len(feature)))
length = len(feature)
for i, word in enumerate(X):
    for j,char in enumerate(word):
        matrix[i,j*length+dictionary[char.lower()]]=1
param_dist = {'objective':'binary:logistic', 'n_estimators':1000}
clf = xgb.XGBClassifier(**param_dist)
x_train, x_test, y_train, y_test = train_test_split(matrix,y ,test_size = 0.2,random_state=1)
search = clf.fit(x_train[:], y_train[:])
y_pred = search.predict(x_test)
print (f1_score(y_test,y_pred))
data = pd.read_csv('test.csv', sep=',',header=0)
X = data['Sequence']
matrix = np.zeros((len(X),4*len(feature)))
for i, word in enumerate(X):
    for j,char in enumerate(word):
        matrix[i,j*length+dictionary[char.lower()]]=1
y_pred=search.predict(matrix)
label_pd = pd.DataFrame(data = y_pred)
label_pd.to_csv('label.csv',header=False,index=False)
