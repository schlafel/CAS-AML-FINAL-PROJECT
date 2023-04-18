import importlib

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import yaml
import argparse
from config import *
from data.data_utils import create_data_loaders_TF
from data.dataset import ASL_DATASET_TF
from sklearn.metrics import classification_report

# tf.config.experimental_run_functions_eagerly(True)


def run_training(PATH_TRAINING_CONFIG):
    """
    Runs a machine learning training script using configuration specified in a YAML file.

    :param PATH_TRAINING_CONFIG (str): Path to the YAML configuration file.
    :return: None
    """

    print(30 * "*", os.path.basename(PATH_TRAINING_CONFIG),30*"*")

    # Load the YAML config file
    with open(PATH_TRAINING_CONFIG, 'r') as f:
        config_model = yaml.safe_load(f)

    ####### Import the Model CLASS #######

    # Import the module and class specified in the YAML file
    module_path = config_model['model']['src']
    class_name = config_model['model']['model_name']
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

    ### Build the model
    model.build([None, *input_shape])
    model.compile(
        loss=config_model["training"]["loss_function"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
        metrics=config_model["training"]["metrics"],
    )

    x = model(np.zeros((1, *input_shape)))
    print(x.shape)
    print(model.summary(expand_nested=True))

    ### Get DataLoaders
    data = ASL_DATASET_TF(augment=config_model["data"]["augment_train"])
    train_dataset,val_dataset,test_dataset = create_data_loaders_TF(data,
                                                                    train_size=TRAIN_SIZE,
                                                                    valid_size=VALID_SIZE,
                                                                    test_size=TEST_SIZE,
                                                                    batch_size = config_model["data"]["batch_size"],
                                                                    augment_train =   config_model["data"]["augment_train"])
    #Log the Datasizes to the conifg_model
    config_model["data"]["TRAIN_SIZE"] = TRAIN_SIZE
    config_model["data"]["VAL_SIZE"] = VALID_SIZE
    config_model["data"]["TEST_SIZE"] = TEST_SIZE
    print(
        f"Training Dataset: {tf.data.experimental.cardinality(train_dataset).numpy()} Elements (Batch-Size {config_model['data']['batch_size']}) --> {config_model['data']['batch_size'] * tf.data.experimental.cardinality(train_dataset).numpy()} samples "
        f"\nValidation Dataset: {tf.data.experimental.cardinality(val_dataset).numpy()} Elements (Batch-Size {config_model['data']['batch_size']}) --> {config_model['data']['batch_size'] * tf.data.experimental.cardinality(val_dataset).numpy()} samples ")




    ############# SETUP PATHS #############
    # TODO: get coorect paths from yaml file (Dynamic Variables)
    if not os.path.exists(config_model["model_dir"]):
        os.makedirs(config_model["model_dir"])

    # Set Experiment Dir
    experiment_dir = os.path.join(config_model["model_dir"], config_model["experiment_name"])
    if os.path.exists(experiment_dir):
        raise Exception(f'Experiment already exists {experiment_dir}')

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Save the YAML configuration file in the model directory
    config_path = os.path.join(experiment_dir, f'config_{config_model["experiment_name"]}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_model, f)

    ############ Callbacks ############

    cb_list = []


    ##       SaveModelCallback
    config_SaveModel = config_model["training"]["callbacks"]["SaveModelCallback"]

    if config_SaveModel["use"]:
        callback_Module = importlib.import_module(config_SaveModel["src"])
        SaveModellCB = getattr(callback_Module, "SaveModelCallback")

        early_stop = SaveModellCB(**config_SaveModel["params"])
        cb_list.append(early_stop)

    ## EarlyStoppingCallback
    config_EarlyStop = config_model["training"]["callbacks"]["EarlyStoppingCallback"]
    if config_EarlyStop["use"]:
        early_stop = tf.keras.callbacks.EarlyStopping(**config_EarlyStop["params"])
        cb_list.append(early_stop)

    # Model ckpt
    mdl_ckpt = config_model["training"]["callbacks"]["ModelCheckpoint"]
    if mdl_ckpt["save"]:
        ckpt_path = os.path.join(experiment_dir, "ckpt")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        mod_ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_path, 'best_model', ),
                **mdl_ckpt["params"]
        )

        cb_list.append(mod_ckpt)

    # Tensorboard log
    config_model["training"]["callbacks"]["TensorBoard"]["log_dir"] = os.path.join(
        config_model["model_dir"], config_model["experiment_name"])

    tensorboard_callback = TensorBoard(
        log_dir=config_model["training"]["callbacks"]["TensorBoard"]["log_dir"],
        histogram_freq=1)
    cb_list.append(tensorboard_callback)

    # # Start training
    # for X_batch, y_batch in train_dataset:
    #     break
    # print(X_batch.shape, X_batch.dtype)


    print("Logging tensorboard to ",
          os.path.abspath(config_model["training"]["callbacks"]["TensorBoard"]["log_dir"] ))
    ############ FIT MODEL ############
    model.fit(train_dataset,
              epochs=config_model["training"]["epochs"],
              callbacks=cb_list,
              validation_data = val_dataset,

              )

    # TODO do the TEST
    ######## TEST Dataset!!
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # Get the total number of samples in the dataset
    num_samples = test_dataset.reduce(0, lambda state, _: state + 1)
    X_test, y_test = test_dataset.map(lambda x, y: (x, y)).unbatch().batch(num_samples).as_numpy_iterator().next()
    y_pred = model.predict(X_test)
    y_pred = tf.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))


    # Save Model to tflite?


    # Load the best model






def main():
    # Clear the TensorFlow session
    tf.keras.backend.clear_session()

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str,
                        default="default_value1", help="Path to the Config YAML-File")

    # Parse arguments
    args = parser.parse_args()

    # Access argument values
    PATH_TRAINING_CONFIG = args.training_config

    # Call the `run_training()` function with the `PATH_TRAINING_CONFIG` argument
    run_training(PATH_TRAINING_CONFIG=PATH_TRAINING_CONFIG)

if __name__ == '__main__':
    main()