import pandas as pd
import numpy as np

label_clf_names = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos",
                   "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2",
                   "LABEL_Bilirubin_direct", "LABEL_EtCO2", "LABEL_Sepsis"]

label_reg_names = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

all_labels = ['pid']
all_labels.extend(label_clf_names)
all_labels.extend(label_reg_names)

f1 = pd.read_csv('submissions/preds/submission_04_22_091550.csv', header=0)
f2 = pd.read_csv('submissions/preds/predictions(1).csv', header=0)

pred1 = f1.values
pred2 = f2.values

pids = pred1[:, 0]

pred = (pred1[:, 1:] + pred2[:, 1:])/2

submiss = np.column_stack([pids, pred])

pd.DataFrame(submiss, columns=all_labels).to_csv('merge.csv', index=False, float_format='%.3f')
