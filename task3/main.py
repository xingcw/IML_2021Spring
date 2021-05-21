import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_data(path, is_train=True):
    df = pd.read_csv(path)
    sequences = df['Sequence'].values
    feats = []
    for seq in sequences:
        num_seqs = [ord(char) - ord('A') for char in seq]
        feats.append(np.asarray(num_seqs))
    if is_train:
        label = df['Active'].values
        return np.asarray(feats), label
    else:
        return np.asarray(feats)


def get_model(input_dim):
    inputs = Input(shape=input_dim)
    # x = Embedding(input_dim=20, output_dim=20, input_length=4)(inputs)
    # x = Flatten()(x)
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # x = Dense(256, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.6)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[x])
    return model


def train():
    # --------------------------preprocessing--------------------------------
    # train_data, train_label = load_data('datasets/train.csv')
    train_data = np.load('datasets/encoded_train_features.npy')
    test_data = np.load('datasets/encoded_test_features.npy')
    train_label = np.load('datasets/oversampled_labels.npy')
    # sampler = RandomUnderSampler()
    # train_data, train_label = sampler.fit_resample(train_data, train_label)
    # one_enc = OneHotEncoder()
    # one_enc.fit(train_data)
    # train_data = one_enc.fit_transform(train_data).toarray()
    # test_data = one_enc.transform(test_data).toarray()
    # scaler = StandardScaler()
    scaler = PowerTransformer()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # ----------------------------training-------------------------------------
    X, X_test, y, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=43)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    test_preds = []
    preds = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        model = get_model(test_data.shape[1])
        model.summary()
        adam = keras.optimizers.Adam(learning_rate=0.001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.001)
        model.compile(adam, loss=tf.keras.losses.binary_crossentropy,
                      metrics=[f1_m, keras.metrics.Precision(), keras.metrics.Recall()])
        model.fit(X_train, y_train, batch_size=1024, epochs=1000, validation_data=(X_val, y_val), shuffle=True,
                  verbose=2, callbacks=[early_stop])
        y_pred = model.predict(X_test)
        test_preds.append(y_pred)
        pred = model.predict(test_data)
        preds.append(pred)
    y_pred = np.mean(np.asarray(test_preds).squeeze(), axis=0)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(classification_report(y_test, y_pred))

    # --------------------------predicting-------------------------------------
    # model.fit(train_data, train_label)
    test_pred = np.mean(np.asarray(preds).squeeze(), axis=0)
    pd.DataFrame(test_pred).to_csv('proba16.csv', index=False, header=None)
    test_pred = np.where(test_pred > 0.5, 1, 0)
    pd.DataFrame(test_pred).to_csv('submission16.csv', index=False, header=None)


if __name__ == '__main__':
    train()
