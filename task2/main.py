import os
import time
import joblib
import pickle
from scipy import stats
from collections import Counter
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, r2_score
from imblearn.under_sampling import RandomUnderSampler


feat_names = ["Age", "EtCO2", "PTT", "BUN", "Lactate", "Temp", "Hgb",
              "HCO3", "BaseExcess", "RRate", "Fibrinogen", "Phosphate", "WBC", "Creatinine",
              "PaCO2", "AST", "FiO2", "Platelets", "SaO2", "Glucose", "ABPm",
              "Magnesium", "Potassium", "ABPd", "Calcium", "Alkalinephos", "SpO2", "Bilirubin_direct",
              "Chloride", "Hct", "Heartrate", "Bilirubin_total", "Troponinl", "ABPs", "pH"]

label_clf_names = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos",
                   "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2",
                   "LABEL_Bilirubin_direct", "LABEL_EtCO2", "LABEL_Sepsis"]

label_reg_names = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

PARAM_DIST = {
    "n_estimators": stats.randint(150, 500),
    "learning_rate": stats.uniform(0.01, 0.07),
    "max_depth": [4, 5, 6, 7],
    "colsample_bytree": stats.uniform(0.5, 0.45),
    "min_child_weight": [1, 2, 3, 4, 5, 6],
    "gamma": [0.1, 0.2, 0.3]
}


def imputation(raw_feats, raw_feats_mean, num_person, rows_per_person):
    imputed_feats = np.zeros_like(raw_feats)
    for person_id in range(num_person):
        start_row_id = person_id * rows_per_person

        # extract feature block from raw feats as new copy
        feats_block = raw_feats[start_row_id: start_row_id + rows_per_person].copy()
        for row_id in range(rows_per_person):
            cur_row = feats_block[row_id]
            nan_idx = np.where(np.isnan(cur_row))
            if row_id == 0:
                cur_row[nan_idx] = raw_feats_mean[nan_idx]
            else:  # 0 < row_id < rows_per_person
                prev_row = feats_block[row_id - 1]
                cur_row[nan_idx] = prev_row[nan_idx]

        imputed_feats[start_row_id: start_row_id + rows_per_person] = feats_block

    return imputed_feats


def preproc(in_feats: np.ndarray, source='train'):
    rows_per_person = 12
    num_row, num_col = in_feats.shape
    num_person = int(num_row / rows_per_person)
    in_feats_mean = np.nanmean(in_feats, axis=0)

    # impute input features
    saved_imputed_path = f"dataset/{source}_feats_imputed.npy"
    if os.path.exists(saved_imputed_path):
        print(f"[Preproc] loading saved imputed data from {saved_imputed_path}")
        in_feats_imputed = np.load(saved_imputed_path)
    else:
        print("[Preproc] start imputing input data")
        in_feats_imputed = imputation(in_feats, in_feats_mean, num_person, rows_per_person)
        print("[Preproc] imputing finished...")
        print(f"[Preproc] saving imputed data to {saved_imputed_path}")
        np.save(saved_imputed_path, in_feats_imputed)

    out_feats = None
    for person_id in range(num_person):  # processing features per person
        feats_this_person_list = []
        # rows from person_id*steps_per_person to (person_id+1)*steps_per_person
        feats_block = in_feats[person_id * rows_per_person: (person_id + 1) * rows_per_person].copy()

        # first feat: last observe value
        last_obs = feats_block[-1]  # last row
        last_obs_nan_idx = np.where(np.isnan(last_obs))
        last_obs[last_obs_nan_idx] = in_feats_mean[last_obs_nan_idx]
        feats_this_person_list.append(last_obs)

        # second feat: number of non-nan values
        block_nan_idx = np.isnan(feats_block)
        num_nan = np.sum(block_nan_idx, axis=0)
        num_not_nan = rows_per_person - num_nan
        feats_this_person_list.append(num_not_nan)

        feats_block_imputed = in_feats_imputed[person_id * rows_per_person: (person_id + 1) * rows_per_person].copy()
        # statistical feat:
        # median
        feats_med = np.median(feats_block_imputed, axis=0)
        feats_this_person_list.append(feats_med)

        # quantile
        feats_quantile = np.quantile(feats_block_imputed, 0.5, axis=0)
        feats_this_person_list.append(feats_quantile)

        feats_this_person = np.hstack(feats_this_person_list)
        if person_id == 0:
            # resize
            out_feats = np.zeros((num_person, feats_this_person.size))
        # append into this row
        out_feats[person_id] = feats_this_person

    # normalize out feats
    normalizer = StandardScaler()
    out_feats = normalizer.fit_transform(out_feats)

    return out_feats


def clf_task(train_data, train_labels):
    for idx, label_name in enumerate(label_clf_names[6:]):
        sampler = RandomUnderSampler()
        train_this_label = train_labels[:, idx]
        train_this_data, train_this_label = sampler.fit_resample(train_data, train_this_label)
        print("=" * 50, '\n')
        print(f"[Classify] classifying {label_name} ")
        print(f"[Classify] {label_name} has {Counter(train_this_label)}")
        time.sleep(5)

        split_frac = 0.9
        split_id = int(split_frac * len(train_this_data))
        shuffle = np.random.permutation(np.asarray(range(0, len(train_this_label))))
        feats, labels = train_this_data[shuffle], train_this_label[shuffle]
        X_train, X_val = feats[:split_id], feats[split_id:]
        y_train, y_val = labels[:split_id], labels[split_id:]

        model = xgb.XGBClassifier(objective='binary:logistic', n_thread=-1)
        clf = RandomizedSearchCV(estimator=model,
                                 param_distributions=PARAM_DIST,
                                 cv=5,
                                 n_iter=50,
                                 scoring="roc_auc",
                                 error_score=0,
                                 verbose=3,
                                 n_jobs=-1, )
        clf.fit(X_train, y_train)
        print(clf.best_estimator_.predict_proba(X_val)[:, 1])
        print(
            f"ROC score on validation set "
            f"{roc_auc_score(y_val, clf.best_estimator_.predict_proba(X_val)[:, 1])}"
        )
        print(f"CV score {clf.best_score_}")
        print(f"Best parameters is {clf.best_params_}")

        joblib.dump(
            clf.best_estimator_,
            f"models/xgboost_fine_{label_name}.pkl",
        )
        time.sleep(5)


def reg_task(train_data, train_labels):
    for idx, label_name in enumerate(label_reg_names):
        train_this_label = train_labels[:, idx]
        print("=" * 50, '\n')
        print(f"[Regression] regressing {idx + 1}th label: {label_name} ")
        time.sleep(5)

        split_frac = 0.9
        split_id = int(split_frac * len(train_data))
        shuffle = np.random.permutation(np.asarray(range(0, len(train_this_label))))
        feats, labels = train_data[shuffle], train_this_label[shuffle]
        X_train, X_val = feats[:split_id], feats[split_id:]
        y_train, y_val = labels[:split_id], labels[split_id:]

        model = xgb.XGBRegressor(objective='reg:squarederror', n_thread=-1)
        reg = RandomizedSearchCV(estimator=model,
                                 param_distributions=PARAM_DIST,
                                 cv=5,
                                 n_iter=50,
                                 scoring="r2",
                                 error_score=0,
                                 verbose=3,
                                 n_jobs=-1, )
        reg.fit(X_train, y_train)
        print(
            f"ROC score on test set "
            f"{r2_score(y_val, reg.best_estimator_.predict(X_val))}"
        )
        print(f"CV score {reg.best_score_}")
        print(f"Best parameters {reg.best_params_}")

        joblib.dump(
            reg.best_estimator_,
            f"models/xgboost_fine_{label_name}.pkl",
        )
        time.sleep(5)


def predict(test_pids, test_feats):
    preds = [test_pids]
    all_labels = ['pid']
    all_labels.extend(label_clf_names)
    all_labels.extend(label_reg_names)
    for idx, label_name in enumerate(label_clf_names):
        model = joblib.load(f'models/xgboost_fine_{label_name}.pkl')
        print(f'[Classification Prediction] {idx}th class {label_name}......')
        pred = model.predict_proba(test_feats)
        preds.append(pred[:, 1])
    for idx, label_name in enumerate(label_reg_names):
        model = joblib.load(f'models/xgboost_fine_{label_name}.pkl')
        print(f'[Regression Prediction] {idx}th label {label_name}......')
        pred = model.predict(test_feats)
        preds.append(pred)
    pd.DataFrame(np.column_stack(preds), columns=all_labels).to_csv('submissions/xgb_preds.csv', index=False,
                                                                    float_format='%.3f')


def main():

    # ---------------------load data-------------------------------
    print("Load dataset.......")
    train_feats = pd.read_csv('dataset/train_features.csv', header=0).values
    train_feats = train_feats[:, 2:]
    test_feats = pd.read_csv('dataset/test_features.csv', header=0).values
    test_pids = test_feats[:, 0].reshape(-1, 12)[:, 0]
    test_feats = test_feats[:, 2:]
    train_labels = pd.read_csv('dataset/train_labels.csv', header=0).values
    train_labels = train_labels[:, 1:]
    train_labels_clf = train_labels[:, :11]
    train_labels_reg = train_labels[:, 11:]

    # ---------------------extract features-------------------------
    train_feats_extracted = preproc(train_feats, source='train')
    test_feats_extracted = preproc(test_feats, source='test')

    # ---------------------make dirs--------------------------------
    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    if not os.path.exists('models'):
        os.makedirs('models')

    # ----------------------train models----------------------------
    clf_task(train_feats_extracted, train_labels_clf)
    reg_task(train_feats_extracted, train_labels_reg)

    # ----------------------prediction-------------------------------
    predict(test_pids, test_feats_extracted)


if __name__ == '__main__':
    main()
