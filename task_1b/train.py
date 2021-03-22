from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('datasets/train.csv', index_col=0).to_numpy()
y = data[:, 0]
X = data[:, 1:]
X = np.concatenate([X, np.square(X), np.exp(X), np.cos(X), np.ones(shape=(X.shape[0], 1))], axis=1)
print('the size of featrues is: {}'.format(X.shape))

scores = []
# alphas = [0.001, 0.1, 1, 10, 100]
alphas = np.arange(10, 30, 1).tolist()
weights = np.zeros(shape=(X.shape[1], len(alphas)))
split = KFold(n_splits=10, random_state=0, shuffle=True)

plt.figure(figsize=(24, 8))
plt.subplot(121)

for i, alpha in enumerate(alphas):
    each_score = []
    each_coef_ = []
    for train_index, test_index in split.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = Ridge(alpha=alpha, solver='sag', fit_intercept=False).fit(X_train, y_train)
        each_coef_.append(model.coef_)
        y_pred = model.predict(X_test)
        each_score.append(np.sqrt(np.mean(np.square(y_pred-y_test), axis=0)))
    weights[:, i] = np.mean(np.asarray(each_coef_), axis=0)
    plt.plot(np.arange(weights.shape[0]), weights[:, i], label='lambda_{:4f}'.format(alpha))
    scores.append(np.mean(each_score))

plt.legend()
plt.xlabel('weight index')

plt.subplot(122)
plt.plot(alphas, scores, label='scores')
plt.legend()
plt.xlabel('lambda')

plt.show()

pd.DataFrame(data=np.asarray(scores)).to_csv('datasets/scores.csv', index=None, header=False)
pd.DataFrame(data=np.asarray(weights)).to_csv('datasets/weights.csv', index=None, header=False)

best_model = Ridge(alpha=300, fit_intercept=False).fit(X, y)
pd.DataFrame(data=best_model.coef_).to_csv('datasets/submission1.csv', index=None, header=False)
# best_model = Lasso(alpha=0.0035).fit(X, y)
pd.DataFrame(data=np.asarray(np.mean(weights, axis=1))).to_csv('datasets/submission.csv', index=None, header=False)