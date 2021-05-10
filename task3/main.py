#main.py

import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from datetime import datetime

# ---------------------Specify Directory-------------------------------
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path,'datasets')
model_path = os.path.join(current_path,'models')

PARAM_DIST = {
    'alpha': np.logspace(-10, 2, num=10),
    'hidden_layer_sizes': [
        (90, 10), (90, 50), (90, 90), (90, 30, 10), (90, 90, 30, 10)
    ]
}

def readBLOSUM(index = 62):
    '''read BLOSUM matrix to DataFrame by calling its index. eg: 50, 62. Type readBLOSUM() for help
    index: int
        The index of BLOSUM matrix, 62 for blosum62
    return DataFrame
        The BLOSUM matrix DataFrame

    '''
    matricesList = [i for i in [40, 50, 55, 60 ,62, 65, 70, 75, 80, 85, 90]]
    if index == 0:
        print("Read BLOSUM matrix as DataFrame.\nAvailable index:", matricesList)
        return 
    filepath = os.path.join(data_path, "BLOSUM"+str(index)+".txt")
    # header = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
    blosum = pd.read_csv(filepath, header=0, index_col=0)
    return blosum

def encode(seq):
    '''
    Encode protein sequence, seq, to one-dimension array.
    Use blosum matrix to encode the number.
    input: [string] seq (length = n)
    output: [1x24n ndarray] e
    '''
    #encode a peptide into blosum features
    matix = readBLOSUM()
    s=list(seq)
    x = pd.DataFrame([matix[i] for i in seq]).reset_index(drop=True)
    e = x.to_numpy().flatten() 
    # print(x)   
    return e

def main():
    # --------------------------Data Importing--------------------------------
    print("Loading dataset .......")
    train_data = pd.read_csv(os.path.join(data_path,'train.csv')).to_numpy()
    train_features = train_data[:,0] #(112000,)
    train_labels = train_data[:,1] #{0: 107787, 1: 4213}
    train_labels = train_labels.astype('int')
    test_data = pd.read_csv(os.path.join(data_path,'test.csv')).to_numpy() #(48000, 1)
    sampler = RandomOverSampler()
    train_features_res, train_labels_res = sampler.fit_resample(train_features.reshape(-1,1), train_labels.reshape(-1,1))
    print("Training labels ... ", Counter(train_labels_res))

    # --------------------------Preprocessing--------------------------------
    saved_Train_data_path = os.path.join(data_path,"encoded_train_features.npy")
    saved_Test_data_path = os.path.join(data_path,"encoded_test_features.npy")
    if os.path.exists(saved_Train_data_path) and os.path.exists(saved_Test_data_path):
        print("Preprocessing ... loading saved encoded training data from"+saved_Train_data_path)
        train_features_res_enc = np.load(saved_Train_data_path)
        print("Preprocessing ... loading saved encoded testing data from"+saved_Test_data_path)
        test_data_enc = np.load(saved_Test_data_path)
    else:
        train_features_res_string = ''.join(train_features_res.flatten())
        print("Preprocessing ... Encoding training data using BLOSUM62 matrix")
        train_features_res_enc = encode(train_features_res_string)
        train_features_res_enc = train_features_res_enc.reshape(-1,96)
        print("Preprocessing ... saving encoded training data to"+saved_Train_data_path)
        train_features_res_enc = np.vstack(train_features_res_enc) #(215574, 96)
        np.save(saved_Train_data_path, train_features_res_enc)

        test_data_string = ''.join(test_data.flatten())
        print("Preprocessing ... Encoding testing data using BLOSUM62 matrix")
        test_data_enc = encode(test_data_string)
        test_data_enc = test_data_enc.reshape(-1,96)
        print("Preprocessing ... saving encoded testing data to"+saved_Test_data_path)
        test_data_enc = np.vstack(test_data_enc) #(48000, 96)
        np.save(saved_Test_data_path, test_data_enc)

    # --------------------------Cross Validation--------------------------------
    model = MLPClassifier(learning_rate='adaptive', max_iter=5000, random_state=42)
    clf = GridSearchCV(estimator=model,
                        param_grid=PARAM_DIST,
                        cv=5,
                        scoring="f1",
                        verbose=5,
                        n_jobs=-1, )
    clf.fit(train_features_res_enc, train_labels_res)
    print("CV score ", clf.best_score_)
    print("Best parameters is ", clf.best_params_)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%H-%M-%S")
    filename = "MLPclassifier-"+current_time+".pkl"
    # --------------------------Refitting--------------------------------
    print("Refitting ... model is stored in ", os.path.join(model_path, filename))
    clf.best_estimator_.fit(train_features_res_enc, train_labels_res)
    joblib.dump(clf.best_estimator_, os.path.join(model_path, filename))

    # --------------------------Predicting--------------------------------
    loadfile = filename
    if os.path.exists(os.path.join(model_path, loadfile)):
        clf_model = joblib.load(os.path.join(model_path, loadfile))
        pred = clf_model.predict(test_data_enc)
        result = pd.DataFrame(pred)
        result.to_csv(os.path.join(data_path,"predictions.csv"), index=False, header=False)

main()