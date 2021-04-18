import json
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.under_sampling import RandomUnderSampler
from RESNET1D import build_network


def class_task(train_feats, train_labels, val_feats, val_labels, test_clf, **config):
    model = build_network(**config)
    model.summary()
    adam = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min', min_delta=0.005)
    model.compile(adam, loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.AUC()])
    model.fit(train_feats, train_labels, batch_size=400, epochs=100, validation_data=(val_feats, val_labels), verbose=2,
              callbacks=[early_stop])
    pred = model.predict(test_clf)
    return pred


def reg_task(train_feats, train_labels, val_feats, val_labels, test_reg, **config):
    model = build_network(**config)
    model.summary()
    adam = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min', min_delta=0.002)
    model.compile(adam, loss=tf.keras.losses.mean_squared_error, metrics=[r2_score])
    model.fit(train_feats, train_labels, batch_size=400, epochs=200, validation_data=(val_feats, val_labels), verbose=2,
              callbacks=[early_stop])
    score = model.evaluate(val_feats, val_labels, verbose=0)
    pred = model.predict(test_reg)
    return score


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.7
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def down_sample(feats, labels):
    feats_0, labels_0 = feats[labels == 0], labels[labels == 0]
    feats_1, labels_1 = feats[labels == 1], labels[labels == 1]
    cnt = Counter(labels)
    if cnt[0] < cnt[1]:
        mask = np.random.permutation(np.argwhere(labels == 1))[:cnt[0]]
        feats_1, labels_1 = feats[mask], labels[mask]
    else:
        mask = np.random.permutation(np.argwhere(labels == 0))[:cnt[1]]
        feats_0, labels_0 = feats[mask], labels[mask]
    feats = np.row_stack([feats_0.squeeze(), feats_1.squeeze()])
    labels = np.concatenate([labels_0.squeeze(), labels_1.squeeze()])
    print(f'{len(feats)} after down sample')
    shuffle = np.random.permutation(np.asarray(range(0, len(labels))))
    feats, labels = feats[shuffle], labels[shuffle]
    return feats, labels


def main():
    train = np.load('dataset/train.npy')
    test = np.load('dataset/test.npy')
    with open('config.json') as f:
        params = json.load(f)

    # ---------------------normalization & reshape-------------------------
    train_pids, train_times, train_features = train[:, 0], train[:, 1], train[:, 2:]
    test_pids, test_times, test_features = test[:, 0], test[:, 1], test[:, 2:]
    num_feats = train_features.shape[1]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.fit_transform(test_features)
    labels_pd = pd.read_csv('dataset/train_labels.csv')
    all_labels, headers = labels_pd.values, labels_pd.columns

    # ---------------------classification task-----------------------------
    train_clf = train_features.reshape(-1, 12 * num_feats)
    test_clf = test_features.reshape(-1, 12, num_feats)
    clf_labels = all_labels[:, 1:12]
    sampler = RandomUnderSampler(random_state=42)
    clf_feats, clf_labels = sampler.fit_resample(train_clf, clf_labels[:, 0])
    clf_feats = clf_feats.reshape(-1, 12, num_feats)
    shuffle = np.random.permutation(np.asarray(range(0, len(clf_labels))))
    feats, labels = clf_feats[shuffle], clf_labels[shuffle]
    # feats, labels = down_sample(train_features, all_labels[:, 1])

    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(feats, labels):
        X_train, X_val = feats[train_index], feats[test_index]
        y_train, y_val = labels[train_index], labels[test_index]
        pred_clf = class_task(X_train, y_train, X_val, y_val, **params)
        np.save('dataset/clf_pred.npy', pred_clf)

    # ---------------------regression task-----------------------------
    train_reg = train_features.reshape(-1, 12, num_feats)
    test_reg = test_features.reshape(-1, 12, num_feats)
    reg_labels = all_labels[:, 12:]
    shuffle = np.random.permutation(np.asarray(range(0, len(reg_labels))))
    feats, labels = train_reg[shuffle], reg_labels[shuffle]

    skf = KFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(feats, labels):
        X_train, X_val = feats[train_index], feats[test_index]
        y_train, y_val = labels[train_index, 2], labels[test_index, 2]
        pred_reg = reg_task(X_train, y_train, X_val, y_val, test_reg, **params)
        np.save('dataset/clf_pred.npy', pred_reg)


if __name__ == '__main__':
    main()
