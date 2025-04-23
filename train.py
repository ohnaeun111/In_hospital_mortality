import os, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.over_sampling import SMOTE
from model import define_model
from evaluation import get_clf_eval
from sklearn.metrics import roc_curve

def find_best_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    J = tpr + (1 - fpr) - 1
    best_idx = np.argmax(J)
    return thresholds[best_idx]

model = define_model()
str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
SavePath = './DNN_best2/'
fold_root = './DNN_best2/folds'
os.makedirs(SavePath, exist_ok=True)

for fold in range(1, 6):
    print(f'--------------------{fold}번째 KFold-------------------')
    fold_path = os.path.join(fold_root, f'fold{fold}')
    data_train = pd.read_csv(os.path.join(fold_path, 'X_train.csv')).values
    label_train = pd.read_csv(os.path.join(fold_path, 'y_train.csv'))['label'].values
    data_valid = pd.read_csv(os.path.join(fold_path, 'X_valid.csv')).values
    label_valid = pd.read_csv(os.path.join(fold_path, 'y_valid.csv'))['label'].values

    smote = SMOTE()
    data_train_over, label_train_over = smote.fit_resample(data_train, label_train)

    callback_list = [
        EarlyStopping(monitor='val_loss', mode='min', patience=10),
        ModelCheckpoint(filepath=SavePath + f'model_fold{fold}.h5', monitor='val_loss', mode='min', save_best_only=True),
    ]
    _ = gc.collect()
    tf.keras.backend.clear_session()
    model = define_model()

    model.fit(data_train_over, label_train_over, validation_data=(data_valid, label_valid),
              batch_size=8, epochs=1000, callbacks=callback_list, verbose=2)

    model.load_weights(SavePath + f'model_threshold1_fold{fold}.h5')
    valid_prob = model.predict(data_valid).squeeze()
    best_threshold = find_best_threshold(label_valid, valid_prob)
    valid_pred = (valid_prob >= best_threshold).astype(int)

    val_metrics = get_clf_eval(label_valid, valid_pred, valid_prob)
    print(f'[Fold {fold} Valid Eval] {val_metrics}')
