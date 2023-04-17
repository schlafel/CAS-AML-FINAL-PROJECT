import importlib
import torch

import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import yaml
import argparse

# tf.config.experimental_run_functions_eagerly(True)
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

    input_shape = tuple(config_model["model"]["params"]["input_shape"])

    model = model_class(**config_model['model']["params"])

    # Set up the learning rate scheduler
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config_model['training']['learning_rate'],
        decay_steps=config_model['training']['decay_steps'],
        decay_rate=config_model['training']['decay_rate'],
        staircase=config_model['training']['staircase']
    )

    ###Build the model
    model.build([None, *input_shape])
    model.compile(
        loss=config_model["training"]["loss_function"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
        metrics=config_model["training"]["metrics"],
    )


    x = model(np.zeros((1, *input_shape)))
    print(x.shape)
    print(model.summary(expand_nested=True))

    # Get the data
    dm = 1

    # Setup Model Dir
    # TODO: get coorect paths from yaml file (Dynamic Variables)
    if not os.path.exists(config_model["model_dir"]):
        os.makedirs(config_model["model_dir"])

    # Define Callbacks
    tf_dataModule = importlib.import_module(config_model["data"]["type"])
    dataset = tf_dataModule.get_tf_dataset(csv_path=config_model["train_csv_file"],
                                           data_path=config_model["data_dir"],
                                           batch_size=config_model["data"]["batch_size"])

    print("got data")

    # Set Experiment Dir
    experiment_dir = os.path.join(config_model["model_dir"], config_model["experiment_name"])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Save the YAML configuration file in the model directory
    config_path = os.path.join(experiment_dir, f'config_{config_model["experiment_name"]}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_model, f)

    # Setup Callbacks
    callback_Module = importlib.import_module(config_model["training"]["callbacks"]["src"])
    cb_list = []
    for callback_name in config_model["training"]["callbacks"]["callbacks"].items():
        if callback_name[0] == "TensorBoard":
            continue
        print(callback_name[0])
        # TODO: Hack: overwrite model_path with model-Dir as dynamic variables do not work somehow....
        if "model_path" in callback_name[1].keys():
            callback_name[1]["model_path"] = experiment_dir

        cb = getattr(callback_Module, callback_name[0])
        cb_list.append(cb(**callback_name[1]))

    config_model["training"]["callbacks"]["callbacks"]["TensorBoard"]["log_dir"] = os.path.join(
        config_model["model_dir"], config_model["experiment_name"])

    tensorboard_callback = TensorBoard(
        log_dir=config_model["training"]["callbacks"]["callbacks"]["TensorBoard"]["log_dir"],
        histogram_freq=1)
    cb_list.append(tensorboard_callback)

    # Start training
    for X_batch, y_batch in dataset:
        break
    print(X_batch.shape, X_batch.dtype)

    model.fit(dataset,
              epochs=config_model["training"]["epochs"],
              callbacks=cb_list,

              )

    # Save Model to tflite?
