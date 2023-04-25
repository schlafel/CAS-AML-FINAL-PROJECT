import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from tensorflow.lite.python.interpreter import Interpreter
from src.config import *
import os
from src.data.data_utils import load_relevant_data_subset
from tqdm import tqdm

from src.models.preprocessing_layers import PreprocessLayer

def load_full_train_datafile():
    # Path to raw json dictionary file
    map_json_file_path = os.path.join(ROOT_PATH, RAW_DATA_DIR, MAP_JSON_FILE)

    # Read the Mapping JSON file to map target values to integer indices
    with open(map_json_file_path) as f:
        label_dict = json.load(f)

    # Path to raw training metadata file
    train_csv_file_path = os.path.join(ROOT_PATH, RAW_DATA_DIR, TRAIN_CSV_FILE)

    # Read the Metadata CVS file for training data
    df_train = pd.read_csv(train_csv_file_path)
    df_train["target"] = df_train.sign.map(label_dict)

    df_train["filepath"] = df_train.path.apply(get_lm_path)


    label_dict_inv = {v: k for k, v in label_dict.items()}

    return df_train,label_dict,label_dict_inv



def get_tflite_model(model, preprocess_layer):

    # TFLite model for submission
    class TFLiteModel(tf.Module):
        def __init__(self, model):
            super(TFLiteModel, self).__init__()

            # Load the feature generation and main models
            self.preprocess_layer = preprocess_layer
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
        def __call__(self, inputs):
            # Preprocess Data
            x, non_empty_frame_idxs = self.preprocess_layer(inputs)
            # tf.print(x.shape)

            # x = x[:,:,:2]
            x = tf.slice(x,[0,0,0],[32,96,2])

            # Add Batch Dimension
            x = tf.expand_dims(x, axis=0)

            # non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
            non_empty_frame_idxs = tf.cast(tf.expand_dims(non_empty_frame_idxs,axis = 0),tf.float32)

            # Make Prediction
            # outputs = self.model({'frames': x, 'non_empty_frame_idxs': non_empty_frame_idxs})
            outputs = self.model(
                (x,non_empty_frame_idxs))
            # Squeeze Output 1x250 -> 250
            outputs = tf.squeeze(outputs, axis=0)

            # Return a dictionary with the output tensor
            return {'outputs': outputs}


    # Define TF Lite Model
    tflite_keras_model = TFLiteModel(model)
    return tflite_keras_model

def get_lm_path(path):
    return os.path.join(ROOT_PATH,RAW_DATA_DIR,path)

def main(path_model,verbose = False ,total = 1000):


    # Load Data

    df_train,label_dict,label_dict_inv = load_full_train_datafile()




    ## Load Model
    model = tf.keras.models.load_model(path_model)

    prep_layer = PreprocessLayer(seq_length=INPUT_SIZE,
                                 useful_landmarks=USEFUL_ALL_LANDMARKS.tolist(),
                                 hand_idxs=USEFUL_HAND_LANDMARKS.tolist())
    x,idx = prep_layer(np.zeros((1,543,3)))

    tflite_keras_model = get_tflite_model(model,prep_layer)


    # Test the predictions



    # prediction before preparation
    correct = 0

    for i in tqdm(range(total)):
        if verbose:
            print(40*"*")
        idx = np.random.randint(0, len(df_train))
        sample = df_train.iloc[idx]
        arr = load_relevant_data_subset(sample.filepath, cols_to_use=["x", "y", "z"])

        if verbose:
            print(sample.filepath)

        # load a sample file
        smpl_file_path = f"train_landmark_files/{sample.participant_id}-{sample.sequence_id}.npz"
        data = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, smpl_file_path))
        lm = data["landmarks"]
        idxs = data["non_empty_idx"]
        data.close()
        preds = model(
            (lm[tf.newaxis, ...],
             tf.cast(idxs[tf.newaxis, ...], tf.float32),
             )
        ).numpy().argmax()
        if verbose:
            print(f"True sign {sample.sign} ({sample.target})")
            print(f"Predicted sign Original Model: {label_dict_inv[preds]} ({preds})")

        x, idx = prep_layer(arr)
        preds_prepModel = tflite_keras_model(arr)["outputs"].numpy().argmax()
        if verbose:
            print(f"Predicted sign by TF-Lite: {label_dict_inv[preds_prepModel]} ({preds_prepModel})")

        if preds_prepModel == sample.target :
            correct+=1
    print(f"Total correct predictions: {correct} ({correct/total*100}%)")


    # Export the Model

    # Create Model Converter
    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)

    keras_model_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]


    # Convert Model
    tflite_model = keras_model_converter.convert()
    # Write Model
    tflite_model_file_path = os.path.join(ROOT_PATH, path_model, "../..", "LSTM_MODEL.tflite")
    with open(tflite_model_file_path, 'wb') as f:
        f.write(tflite_model)







if __name__ == '__main__':
    path_model = os.path.join(ROOT_PATH, MODEL_DIR,
                              r"TF_LSTM_BASELINE/Exp_04/ckpt/best_model")
    main(path_model ,total = 100,verbose=False)


