"""main.py
tensorflow=2.5.0
opencv-python=4.5.2
"""
import os
from sys import argv
from datetime import datetime

import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications.xception import Xception, preprocess_input as Xception_preproc
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as ResNet50_preproc
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input as ResNet101_preproc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as ResNet50V2_preproc
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as MobileNetV2_preproc
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as InceptionV3_preproc

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# ---------------------Specify Directory-------------------------------
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path,'datasets')
image_path = os.path.join(data_path,'food')
model_path = os.path.join(current_path,'models')
prediction_path = os.path.join(current_path,'predictions')

# ---------------------Configurations----------------------------------
TASK = 'assembly'
BASE_MODEL_NAME = "ResNet101" #argv[1]
IF_AUG = False #argv[2]
START_TIME = "06-01-18-08-04"
ASSEMBLY_NAMES = ["06-01-18-08-04", "06-01-20-18-43"]
N_HIDDEN_LAYERS = 2
N_HIDDEN_UNITS = [1000, 500]
ACTI_FUNC = ['relu', 'relu']
DROP_FRAC = [0.8, 0.8]
BATCH_SIZE = 128
LEARN_RATE = 1e-3
N_FOLDDS = 5

MODELS = {'Xception':[Xception(weights='imagenet', pooling='avg', include_top=False), Xception_preproc], 
        'ResNet50': [ResNet50(weights='imagenet', pooling='avg', include_top=False), ResNet50_preproc], 
        'ResNet50V2': [ResNet50V2(weights='imagenet', pooling='avg', include_top=False), ResNet50V2_preproc], 
        'MobileNetV2': [MobileNetV2(weights='imagenet', pooling='avg', include_top=False), MobileNetV2_preproc], 
        'InceptionV3': [InceptionV3(weights='imagenet', pooling='avg', include_top=False), InceptionV3_preproc], 
        'ResNet101': [ResNet101(weights='imagenet', pooling='avg', include_top=False), ResNet101_preproc]}

# --------------------- Functions ---------------------------------------
def extract(model_name, img_height=224, img_width=224, batch_size=200):
    #read and preprocess images in batches because of memory limit

    print("Preprocessing ... Using model ...", model_name)
    model = MODELS[model_name][0]
    preprocess_input = MODELS[model_name][1]
    
    # directory for saving embed
    extract_sub_dir = 'extract'
    save_dir = os.path.join(data_path, model_name, extract_sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_batch = None
    # Read and preprocess images in batches
    FileList = sorted(os.listdir(image_path))
    n_FileList = len(FileList)
    n_batch = n_FileList//batch_size
    for i in range(n_batch):
        print("Preprocessing ... Extract %dth-%dth image" %(i*batch_size, (i+1)*batch_size))
        image_batch = []
        for j in range(batch_size):
            index = i*batch_size + j
            filename = os.fsdecode(FileList[index])
            # image = cv2.imread(os.path.join(image_path, filename))
            # x = cv2.resize(image, (img_width, img_height))
            img = load_img(os.path.join(image_path, filename), target_size=(img_height, img_width))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_batch.append(x)

        img_batch = np.concatenate(image_batch, axis=0)
        feats = model(img_batch)
        feats_np = feats.numpy().reshape(batch_size, -1)
        print(feats_np.shape)
        # save embedding
        save_path = os.path.join(save_dir, f"batch_{i:0>2d}.npy")
        print("Preprocessing ... Save batch images to ", save_path)
        np.save(save_path, feats_np)

    # plus the remained batch
    if (n_FileList>n_batch*batch_size):
        len_lastBatch = n_FileList-n_batch*batch_size
        image_batch = []
        for i in range(len_lastBatch):
            index = n_batch*batch_size+i
            filename = os.fsdecode(FileList[index])
            image = cv2.imread(os.path.join(image_path, filename))
            x = cv2.resize(image, (img_width, img_height))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_batch.append(x)
        img_batch = np.concatenate(image_batch, axis=0)
        feats = model(img_batch)
        feats_np = feats.numpy().reshape(batch_size, -1)
        print(feats_np.shape)
        # save embedding
        save_path = os.path.join(save_dir, f"batch_{(n_batch+1):0>2d}.npy")
        print("Preprocessing ... Save batch images to ", save_path)
        np.save(save_path, feats_np)

def train_model(base_model_name, train_triplets, n_hidden_layers, n_hidden_units, 
                act_func, drop_frac, batch_size=128, lr=1e-3, early_stop_iters=3, n_folds=5):
    
    print("Training ... using model ", base_model_name)
    # Check parameter setting
    assert n_hidden_layers == len(n_hidden_units), "Check n_hidden_units"
    assert n_hidden_layers == len(act_func), "Check act_func"
    assert n_hidden_layers == len(drop_frac), "Check drop_frac"

    # Read preprocessed data
    extract_sub_dir = 'extract'
    extract_dir = os.path.join(data_path, base_model_name, extract_sub_dir)

    image_batch_list = []
    for file in sorted(os.listdir(extract_dir)):
        filename = os.fsdecode(file)
        cur_arr = np.load(os.path.join(extract_dir, filename))
        image_batch_list.append(cur_arr)
    total_images = np.concatenate(image_batch_list, axis=0) #(10000, 2048)

    del image_batch_list  # save memory

    pos_arr = np.zeros((train_triplets.shape[0], 3 * total_images.shape[1]))  # positive data (59515, 6144)
    neg_arr = np.zeros((train_triplets.shape[0], 3 * total_images.shape[1]))  # negative data, revert order of Pos and Neg

    print("Generating training data ...")
    # Iterate every triplet
    for row_idx in range(train_triplets.shape[0]):
        cur_pos_triplet = train_triplets[row_idx, :].copy()
        cur_neg_triplet = train_triplets[row_idx, :].copy()

        # revert the order of A and B
        cur_neg_triplet[1] = cur_pos_triplet[2]
        cur_neg_triplet[2] = cur_pos_triplet[1]

        pos_arr[row_idx] = total_images[cur_pos_triplet].ravel()  # (6144,)
        neg_arr[row_idx] = total_images[cur_neg_triplet].ravel()  # (6144,)

    pos_label = np.ones(pos_arr.shape[0])
    neg_label = np.zeros(neg_arr.shape[0])

    total_data = np.concatenate([pos_arr, neg_arr], axis=0).astype(np.float32)
    total_label = np.concatenate([pos_label, neg_label], axis=0).astype(np.int8)

    del pos_arr, neg_arr, pos_label, neg_label, total_images, cur_arr, train_triplets  # save memory

    # shuffle dataset
    total_data, total_label = shuffle(total_data, total_label, random_state=42)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%H-%M-%S")
    save_dir = os.path.join(model_path, base_model_name, current_time)
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds)
    fold_idx = 0
    acc_list = []
    for train_idx, val_idx in skf.split(total_data, total_label):
        print("Training ... fold ", fold_idx)
        X_train, X_val = total_data[train_idx], total_data[val_idx]
        y_train, y_val = total_label[train_idx], total_label[val_idx]

        # Construct CNN
        model = Sequential()

        model.add(InputLayer(input_shape=(total_data.shape[1], )))
        for layer_id in range(n_hidden_layers):
            model.add(Dense(n_hidden_units[layer_id]))
            model.add(BatchNormalization())
            model.add(Activation(act_func[layer_id]))
            model.add(Dropout(drop_frac[layer_id]))
        # classify
        last_act_func = 'sigmoid'
        model.add(Dense(1, activation=last_act_func))

        optimizer = Adam(learning_rate = lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        model.summary()

        early_cb = EarlyStopping(monitor='val_loss', patience=early_stop_iters)

        model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=100,
                callbacks=[early_cb],
                validation_data=(X_val, y_val)
        )

        model.save(os.path.join(save_dir, f"fold_{fold_idx}.h5"))

        y_val_pred = model.predict(X_val)
        y_val_pred = np.round(y_val_pred).ravel() 

        acc_score = accuracy_score(y_val, y_val_pred)
        print("Fold %d accuracy: " %fold_idx, acc_score)
        acc_list.append(acc_score)

        fold_idx += 1

    avg_acc_score = sum(acc_list) / len(acc_list)
    acc_df = pd.DataFrame(acc_list)
    print(acc_df)
    print(current_time, " Avg Accuracy = ", avg_acc_score)

    # Re-fitting
    print("Training ... Refitting ...")
    refit_model = Sequential()
    refit_model.add(InputLayer(input_shape=(total_data.shape[1], )))
    for layer_id in range(n_hidden_layers):
        refit_model.add(Dense(n_hidden_units[layer_id]))
        refit_model.add(BatchNormalization())
        refit_model.add(Activation(act_func[layer_id]))
        refit_model.add(Dropout(drop_frac[layer_id]))
    # classify
    last_act_func = 'sigmoid'
    refit_model.add(Dense(1, activation=last_act_func))
    optimizer = Adam(learning_rate=lr)
    refit_model.compile(optimizer=optimizer, loss='binary_crossentropy')
    refit_model.summary()
    refit_model.fit(total_data, total_label,
            batch_size=batch_size,
            epochs=200,
    )
    model.save(os.path.join(save_dir, "refit.h5"))

def predict(base_model_name, test_triplets, n_folds, model_filename):
    print("Predicting ... with ", base_model_name, " from ", model_filename)
    model_dir = os.path.join(model_path, base_model_name, model_filename)

    # directly generate, save disk space
    extract_dir = os.path.join(data_path, base_model_name, 'extract')

    image_batch_list = []
    for file in sorted(os.listdir(extract_dir)):
        filename = os.fsdecode(file)
        cur_arr = np.load(os.path.join(extract_dir, filename))
        image_batch_list.append(cur_arr)

    total_images = np.concatenate(image_batch_list, axis=0)
    test_arr = np.zeros((test_triplets.shape[0], 3 * total_images.shape[1]))

    for row_idx in range(test_triplets.shape[0]):
        cur_triplet = test_triplets[row_idx, :].copy()
        test_arr[row_idx] = total_images[cur_triplet].ravel()

    test_data = test_arr.astype(np.float32)
    del total_images, test_arr  # save memory

    # average 5 fold results
    preds_list = []
    for fold_idx in range(n_folds):
        model = load_model(os.path.join(model_dir, f"fold_{fold_idx}.h5"))
        preds = model.predict(test_data)
        preds_list.append(preds)
    preds_mat = np.column_stack(preds_list)
    row_sum = np.sum(preds_mat, axis=1)
    row_mean = np.mean(preds_mat, axis=1)
    out_preds = np.zeros((test_data.shape[0], ), dtype=np.int8)
    out_preds[row_sum < n_folds/2] = 0
    out_preds[row_sum >= n_folds/2] = 1

    #refit result
    refit_model = load_model(os.path.join(model_dir, "refit.h5"))
    refit_preds = refit_model.predict(test_data)
    refit_out_preds = np.zeros((test_data.shape[0], ), dtype=np.int8)
    refit_out_preds[row_sum < 0.5] = 0
    refit_out_preds[row_sum >= 0.5] = 1

    save_dir = os.path.join(prediction_path, model_filename)
    os.makedirs(save_dir, exist_ok=True)

    save_path_fold = os.path.join(save_dir, "fold.txt")
    np.savetxt(save_path_fold, out_preds.astype(int), fmt='%i')
    print("Predicting ... fold prediction done and saved")

    save_path_fold_prob = os.path.join(save_dir, "fold_prob.txt")
    np.savetxt(save_path_fold_prob, row_mean)
    print("Predicting ... fold probability prediction done and saved")

    save_path_refit = os.path.join(save_dir, "refit.txt")
    np.savetxt(save_path_refit, refit_out_preds.astype(int), fmt='%i')
    print("Predicting ... refit prediction done and saved")

def assembly(pred_names):
    print("assembly results from ", pred_names)
    n_assembly = len(pred_names)
    preds_list = []
    for pred_name in pred_names:
        pred_path = os.path.join(prediction_path, pred_name, 'fold_prob.txt')
        file = np.loadtxt(pred_path)
        preds_list.append(file)
    all_preds = np.column_stack(preds_list)
    print(all_preds)
    out_preds = np.zeros((all_preds.shape[0], ), dtype=np.int8)
    row_sum = np.mean(all_preds, axis=1)
    out_preds[row_sum < 0.5] = 0
    out_preds[row_sum >= 0.5] = 1

    save_path = os.path.join(prediction_path, 'assembly.txt')
    np.savetxt(save_path, out_preds.astype(int), fmt='%i')

def voter():
    path1 = os.path.join(prediction_path, "06-01-18-08-04", "fold.txt")
    path2 = os.path.join(prediction_path, "06-01-20-18-43", "fold.txt")
    path3 = os.path.join(prediction_path, "assembly.txt")

    file1 = np.loadtxt(path1)
    file2 = np.loadtxt(path2)
    file3 = np.loadtxt(path3)

    all_pred = np.column_stack([file1, file2, file3])
    number_1s = np.sum(all_pred, axis=1)
    number_0s = 3 - number_1s
    vote = np.where(number_1s < number_0s, np.zeros_like(number_0s), np.ones_like(number_1s))
    np.savetxt(os.path.join(prediction_path, 'vote.txt'), vote, fmt='%i')

def main():
    # ---------------------load data-------------------------------
    print("\nLoading dataset .......")

    train_raw = pd.read_csv(os.path.join(data_path, "train_triplets.txt"), names=["A", "B", "C"], sep=" ").to_numpy()
    train_raw = train_raw.astype(int)
    test_raw = pd.read_csv(os.path.join(data_path, "test_triplets.txt"), names=["A", "B", "C"], sep=" ").to_numpy()
    test_raw  = test_raw.astype(int)
    print("Dataset loaded ... Shape of traning tripletes: ", train_raw.shape) #(59515, 3)
    print("Dataset loaded ... Shape of testing tripletes: ", test_raw.shape) #(59544, 3)
    
    # ---------------------preprocessing-------------------------------
    if TASK == 'extract':
        print("\nPreprocessing .......")
        extract(model_name=BASE_MODEL_NAME, img_height=224, img_width=224, batch_size=200,)

    # ---------------------training-------------------------------
    elif TASK == 'train':
        print("\nTraining .......")
        train_model(base_model_name=BASE_MODEL_NAME, train_triplets=train_raw,
                    n_hidden_layers=N_HIDDEN_LAYERS,
                    n_hidden_units=N_HIDDEN_UNITS,
                    act_func=ACTI_FUNC,
                    drop_frac=DROP_FRAC,
                    batch_size=BATCH_SIZE, lr=LEARN_RATE,
                    early_stop_iters=3, n_folds=N_FOLDDS)

    # ---------------------predicting-------------------------------
    elif TASK == 'predict':
        print("\nPredicting .......")
        predict(BASE_MODEL_NAME, test_triplets=test_raw, n_folds=N_FOLDDS, model_filename=START_TIME)

    elif TASK == 'assembly':
        print("\nAssembling predicted results .......")
        assembly(ASSEMBLY_NAMES)
        
main()