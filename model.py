import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import losses, optimizers
import scipy.io as sio
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

def eval_model(input_data, SavePath):
    print(f"SavePath: {SavePath}") 
    n_folds = 5
    all_fold_models = []
 
    for num_folds in range(1, n_folds + 1):
        model_path = os.path.join(SavePath, f'model_fold{num_folds}.h5')
        print(f"Trying to load model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: File does not exist - {model_path}")
            continue
        loaded_models= load_model(model_path)
        all_fold_models.append(loaded_models)

    if not all_fold_models:
        raise FileNotFoundError("No models could be loaded. Please check the model paths.")

    test_eval_probas = []

    for model in all_fold_models:
        probas = model.predict(input_data) 
        test_eval_probas.append(probas)

    final_test_proba = np.mean(test_eval_probas, axis=0)

    # 최종 예측 확률 전체를 반환
    return final_test_proba
