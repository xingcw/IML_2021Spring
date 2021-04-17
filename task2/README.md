# Task2 Description

# Test ideas
## preprocessing
we have `missing features`, `imbalanced classification`, and `outliers` in the dataset. Using `group` and `global` imputation to fill missing values, `outlier detection method` to remove outliers, and `resampling method` or `class weight` to deal with imbalance problem.

## Time-series problem
`LSTM` may be useful for extracting features and `SVC` may be used for classification. Or we could build a simple neural network right after the `LSTM` layers.

## results of LSTM
Due to the short length of the time-series and the small number of samples after undersapling, `LSTM` model tends to overfit the training data and gets bad generalization abilities. Besides, the neural networks have too many tuning parameters to tune them well. Then we went for the traditional feature engineering.

# Final Submit Version

## preprocessing
### imputation
We first implement the forward imputation by hand. Namely, we impute missing values of a specific person using the mean of the existing values one by one. If all the values of a specific feature are missing for a specific person, we then impute the missing value by the global mean value of all persons in the dataset. 

### feature extraction
For every original feature of a specific person, we extract 4 features from it, i.e. last observation, number of non-nan values, median, and 0.5 quantile of all 12 values. 

### normalization
We normalize our data using sklearn.preprocessing.StandardScaler.

## Train models
### classification task
We use same method and classifier for subtask1 ans subtask2. We first collect imbalanced information of each label and then undersample all majority samples to make data balanced. Then we apply RandomizedSearchCV to XGBClassifier and pass ranged values for tuning parameters, and then train according to proposed loss and metrics. We predict probability using saved best model.

### regression task
We change model to XGBRegressor and apply same method to get last subtask's prediction.