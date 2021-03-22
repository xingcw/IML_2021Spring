from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


data = pd.read_csv('dataset/train.csv').to_numpy()
y = data[:, 0]
X = data[:, 1:]
print('the size of featrues is: {}'.format(X.shape))

scores = []
alphas = [0.1, 1, 10, 100, 200]
split = KFold(n_splits=10, random_state=42, shuffle=True)


for alpha in alphas:
    each_score = []
    for train_index, test_index in split.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        each_score.append(np.sqrt(np.mean(np.square(y_pred-y_test), axis=0)))
    scores.append(np.mean(each_score))

pd.DataFrame(data=np.asarray(scores)).to_csv('dataset/test.csv', index=None, header=False)
