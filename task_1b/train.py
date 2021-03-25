from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path,'datasets')
data = pd.read_csv(os.path.join(data_path,'train.csv'), index_col=0).to_numpy()
y = data[:, 0]
X = data[:, 1:] #size = (700, 5)
X = np.concatenate([X, np.square(X), np.exp(X), np.cos(X), np.ones(shape=(X.shape[0], 1))], axis=1)
print('the size of featrues is: {}'.format(X.shape)) #(700, 21)

def BestAlphaList(times):
    LeastScoreAlphas = np.zeros(shape = (times,1))
    LeastScores = np.zeros(shape = (times,1))
    for rnd in range(times):
        scores = []
        # alphas = [0.001, 0.1, 1, 10, 100]
        alphas = np.arange(15, 20, 1).tolist()
        weights = np.zeros(shape=(X.shape[1], len(alphas)))
        split = KFold(n_splits=10, random_state=0, shuffle=True)
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
            scores.append(np.mean(each_score))
        least_score = min(scores)
        LeastScores[rnd] = least_score
        LeastScoreAlphas[rnd]  = alphas[scores.index(least_score)]
    AlphaList = pd.DataFrame(np.concatenate([LeastScores,LeastScoreAlphas], axis=1), columns=['score','alpha'])
    print(AlphaList)
    return AlphaList

def FindBestAlpha(): 
    AlphaList = BestAlphaList(1000)
    print(np.mean(AlphaList['alpha']), np.mean(AlphaList['score']))
    least_alphas = AlphaList['alpha'].tolist()
    print({x:least_alphas.count(x) for x in least_alphas})

    plt.figure(figsize=(24, 8))
    counts, bins = np.histogram(AlphaList['alpha'])
    plt.hist(bins[:-1], bins, weights=counts)
    plt.legend()
    plt.show()

def showAlphaEffect(Afrom, Ato):
    scores = []
    # alphas = [0.001, 0.1, 1, 10, 100]
    alphas = np.arange(Afrom, Ato, 1).tolist()
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

    pd.DataFrame(data=np.asarray(scores)).to_csv(os.path.join(data_path,'scores.csv'), index=None, header=False)
    pd.DataFrame(data=np.asarray(weights)).to_csv(os.path.join(data_path,'weights.csv'), index=None, header=False)

def fitBestModel(alp):
    best_model = Ridge(alpha=alp, fit_intercept=False).fit(X, y)
    pd.DataFrame(data=best_model.coef_).to_csv(os.path.join(data_path,'submission1.csv'), index=None, header=False)

# FindBestAlpha()
# showAlphaEffect(10,300)
fitBestModel(325)
