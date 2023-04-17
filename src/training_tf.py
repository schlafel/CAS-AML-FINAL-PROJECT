import importlib
import torch

import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import yaml
import argparse



if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str,
                        default="default_value1", help="Path to the Config YAML-File")


    # Parse arguments
    args = parser.parse_args()

    # Access argument values
    PATH_TRAINING_CONFIG = args.training_config

    # Load the YAML config file
    with open(PATH_TRAINING_CONFIG, 'r') as f:
        config_model = yaml.safe_load(f)


    ####### Import the Model CLASS #######

    # Import the module and class specified in the YAML file
    module_path, class_name = config_model['model']['type'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    input_shape = config_model["model"]["params"]["input_shape"]

    model = model_class(**config_model['model']["params"])
    model.build([None, *input_shape])
    model.compile(loss=config_model["training"]["loss_function"],
                  optimizer=config_model["training"]["optimizer"],
                  metrics = config_model["training"]["metrics"])
    x = model(np.zeros((1, *input_shape)))
    print(x.shape)
    print(model.summary(expand_nested = True))


    # Get the data
    dm = 1

    #Setup Model Dir
    if not os.path.exists(config_model["model_dir"]):
        os.makedirs(config_model["model_dir"])




    # Define Callbacks
    tf_dataModule =  importlib.import_module(config_model["data"]["type"])
    dataset = tf_dataModule.get_tf_dataset(csv_path = config_model["train_csv_file"],
                                           data_path = config_model["data_dir"],
                                           batch_size = config_model["data"]["batch_size"])

    print("got data")
    #Setup Callbacks
    callback_Module = importlib.import_module(config_model["training"]["callbacks"]["src"])
    cb_list = []
    for callback_name in config_model["training"]["callbacks"]["callbacks"].items():
        if callback_name[0] == "TensorBoard":
            continue
        print(callback_name[0])
        cb = getattr(callback_Module, callback_name[0])
        cb_list.append(cb(**callback_name[1]))
    tensorboard_callback = TensorBoard(log_dir=config_model["training"]["callbacks"]["callbacks"]["TensorBoard"]["log_dir"],
                                       histogram_freq=1)
    cb_list.append(tensorboard_callback)


    # Start training
    for X_batch,y_batch in dataset:
        break
    print(X_batch.shape,X_batch.dtype)



    model.fit(dataset,
              epochs = config_model["training"]["epochs"],
              callbacks = cb_list
              )

    # Save Model to tflite?







