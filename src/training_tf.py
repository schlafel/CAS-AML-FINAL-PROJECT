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

    ############# LOAD DATA #############
    tf_dataModule = importlib.import_module(config_model["data"]["type"])
    train_dataset = tf_dataModule.get_tf_dataset(csv_path=config_model["train_csv_file"],
                                                 data_path=config_model["data_dir"],
                                                 batch_size=config_model["data"]["batch_size"],
                                                 augment_data=True)
    val_dataset = tf_dataModule.get_tf_dataset(csv_path=config_model["val_csv_file"],
                                               data_path=config_model["data_dir"],
                                               batch_size=config_model["data"]["batch_size"],
                                               augment_data=False)

    print(
        f"Training Dataset: {tf.data.experimental.cardinality(train_dataset).numpy()} Elements (Batch-Size {config_model['data']['batch_size']}) --> {config_model['data']['batch_size'] * tf.data.experimental.cardinality(train_dataset).numpy()} samples "
        f"\nValidation Dataset: {tf.data.experimental.cardinality(val_dataset).numpy()} Elements (Batch-Size {config_model['data']['batch_size']}) --> {config_model['data']['batch_size'] * tf.data.experimental.cardinality(val_dataset).numpy()} samples ")




    ############# SETUP PATHS #############
    # TODO: get coorect paths from yaml file (Dynamic Variables)
    if not os.path.exists(config_model["model_dir"]):
        os.makedirs(config_model["model_dir"])

    # Set Experiment Dir
    experiment_dir = os.path.join(config_model["model_dir"], config_model["experiment_name"])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Save the YAML configuration file in the model directory
    config_path = os.path.join(experiment_dir, f'config_{config_model["experiment_name"]}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_model, f)

    ############ Callbacks ############
    callback_Module = importlib.import_module(config_model["training"]["callbacks"]["src"])
    cb_list = []
    for callback_name in config_model["training"]["callbacks"]["callbacks"].items():
        if callback_name[0] in ["TensorBoard",
                             "ModelCheckpoint",
                             "EarlyStoppingCallback",
                             ]:
            continue
        # print(callback_name[0])
        # TODO: Hack: overwrite model_path with model-Dir as dynamic variables do not work somehow....
        if "model_path" in callback_name[1].keys():
            callback_name[1]["model_path"] = experiment_dir

        cb = getattr(callback_Module, callback_name[0])
        cb_list.append(cb(**callback_name[1]))


    ##      EarlyStoppingCallback
    config_EarlyStop = config_model["training"]["callbacks"]["callbacks"]["EarlyStoppingCallback"]
    if config_EarlyStop["use"]:
        early_stop = tf.keras.callbacks.EarlyStopping(**config_EarlyStop["params"])

        cb_list.append(early_stop)

    # Model ckpt
    mdl_ckpt = config_model["training"]["callbacks"]["callbacks"]["ModelCheckpoint"]
    if mdl_ckpt["save"]:
        ckpt_path = os.path.join(experiment_dir, "ckpt")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        mod_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(experiment_dir, 'checkpoint-{epoch:02d}.h5', ),
            save_weights_only=True,
            monitor=mdl_ckpt["monitor"],
            mode=mdl_ckpt["mode"],
            save_best_only=False,
            save_freq=mdl_ckpt["save_freq"]
        )

        cb_list.append(mod_ckpt)

    # Tensorboard log
    config_model["training"]["callbacks"]["callbacks"]["TensorBoard"]["log_dir"] = os.path.join(
        config_model["model_dir"], config_model["experiment_name"])

    tensorboard_callback = TensorBoard(
        log_dir=config_model["training"]["callbacks"]["callbacks"]["TensorBoard"]["log_dir"],
        histogram_freq=1)
    cb_list.append(tensorboard_callback)

    # Start training
    for X_batch, y_batch in train_dataset:
        break
    print(X_batch.shape, X_batch.dtype)



    ############ FIT MODEL ############
    model.fit(train_dataset,
              epochs=config_model["training"]["epochs"],
              callbacks=cb_list,
              validation_data = val_dataset,

              )

    # Save Model to tflite?
