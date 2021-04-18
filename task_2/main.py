from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path,'datasets')
train_features = pd.read_csv(os.path.join(data_path,'train_features.csv'), index_col=0).to_numpy()
train_labels = pd.read_csv(os.path.join(data_path,'train_labels.csv'), index_col=0).to_numpy()
test_features = pd.read_csv(os.path.join(data_path,'test_features.csv'), index_col=0).to_numpy()
