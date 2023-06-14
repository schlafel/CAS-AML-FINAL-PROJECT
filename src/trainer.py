"""
===========================
Generic trainer description
===========================

Trainer module handles the training, validation, and testing of framework-agnostic deep learning models.

The Trainer class handles the complete lifecycle of model training including setup, execution of training epochs,
validation and testing, early stopping, and result logging.

The class uses configurable parameters for defining training settings like early stopping and batch size, and it
supports adding custom callback functions to be executed at the end of each epoch. This makes the trainer class flexible
and adaptable for various types of deep learning models and tasks.

Attributes:
model_name (str): The name of the model to be trained.
params (dict): The parameters required for the model.
model (model object): The model object built using the given model name and parameters.
train_loader, valid_loader, test_loader (DataLoader objects): PyTorch dataloaders for training, validation, and testing datasets.
patience (int): The number of epochs to wait before stopping training when the validation loss is no longer improving.
best_val_metric (float): The best validation metric recorded.
patience_counter (int): A counter that keeps track of the number of epochs since the validation loss last improved.
model_class (str): The class name of the model.
train_start_time (str): The starting time of the training process.
writer (SummaryWriter object): TensorBoard's SummaryWriter to log metrics for visualization.
checkpoint_path (str): The path where the best model checkpoints will be saved during training.
epoch (int): The current epoch number.
callbacks (list): A list of callback functions to be called at the end of each epoch.

Methods:
train(n_epochs): Trains the model for a specified number of epochs.
evaluate(): Evaluates the model on the validation set.
test(): Tests the model on the test set.
add_callback(callback): Adds a callback function to the list of functions to be called at the end of each epoch.

"""
import sys,os
from shutil import copyfile
sys.path.insert(0, '../src')

import config as config
import numpy as np
from tqdm import tqdm

import time

import importlib
from torch.utils.tensorboard import SummaryWriter,summary
from dl_utils import get_model_params, log_metrics, log_hparams_metrics,get_metric_dict
from data.data_utils import create_data_loaders
from data.dataset import ASL_DATASET
from datetime import datetime
from callbacks import dropout_callback, augmentation_increase_callback
import yaml
from metrics import Metric
import torch
import inspect
import callbacks

import  warnings
from sklearn.metrics import classification_report
class Trainer:
    """
    A trainer class which acts as a control hub for the model lifecycle, including initial setup, executing training
    epochs, performing validation and testing, implementing early stopping, and logging results. The module has been
    designed to be agnostic to the specific deep learning framework, enhancing its versatility across various projects.
    """
    def __init__(self, config, dataset=ASL_DATASET, model_config = None):
        """
        Initializes the Trainer class with the specified parameters.

        This method initializes various components needed for the training process. This includes the model
        specified by the model name, the dataset with optional data augmentation and dropout, data loaders for
        the training, validation, and test sets, a SummaryWriter for logging, and a path for saving model checkpoints.

        #. The method first retrieves the specified model and its parameters.
        #. It then initializes the dataset and the data loaders.
        #. It sets up metrics for early stopping and a writer for logging.
        #. Finally, it prepares a directory for saving model checkpoints.

        Args:
            config (module): the config with the hyperparameters for model training
            dataset (Dataset): The dataset to be used.
            model_config (optional):



        Functionality:
            This method initializes various components, such as the model, dataset, data loaders,
            logging writer, and checkpoint path, required for the training process.

        :param: config: module
        :type config: module
        :param dataset: The dataset for training.
        :type dataset: Dataset or None
        :param model_config: predifined model configuration
        :type model_config: dict or None


        :rtype: None

        .. note::
            This method only initializes the Trainer class. The actual training is done by calling the train() method.

        .. warning::
            Make sure the specified model name corresponds to an actual model in your project's models directory.
        """
        self.model_name = config.MODELNAME
        self.DL_FRAMEWORK = config.DL_FRAMEWORK
        module_name = f"models.{config.DL_FRAMEWORK}.models"

        #Get Model Params....
        if model_config is None:
            self.params = get_model_params(self.model_name)
        else:
            self.params = model_config.copy()

        module = importlib.import_module(module_name)
        Model = getattr(module, self.model_name)

        # Get the attributes from the config module
        config_items = inspect.getmembers(config)

        # Filter out private attributes and built-in attributes
        self.config_dict = {name: value for name, value in config_items if
                       not name.startswith('_') and not inspect.ismodule(value)}

        # Get Model
        self.model = Model(**self.params)
        print(f"Using model: {module_name}.{self.model_name}")

        # Get Data
        self.dataset = dataset(**self.params['data'])
        self.train_loader, self.valid_loader, self.test_loader = create_data_loaders(
            self.dataset, batch_size=config.BATCH_SIZE, dl_framework=config.DL_FRAMEWORK, num_workers=4)

        batch = next(iter(self.train_loader))[0]
        self.model(batch)

        self.patience = config.EARLY_STOP_PATIENCE
        self.best_val_metric = float('inf') if config.EARLY_STOP_MODE == "min" else float('-inf')
        self.patience_counter = 0

        now = datetime.now()

        self.model_class = self.model.__class__.__name__
        self.train_start_time = now.strftime("%Y-%m-%d %H_%M")

        self.log_dir = os.path.join(config.ROOT_PATH, config.RUNS_DIR, config.DL_FRAMEWORK,
                                                 self.model_class, self.train_start_time)

        self.writer = SummaryWriter(self.log_dir,
                                    filename_suffix="experiment")
        self.metrics = config.LOG_METRICS
        self.checkpoint_path = os.path.join(config.ROOT_PATH, config.CHECKPOINT_DIR, config.DL_FRAMEWORK, self.model_class, self.train_start_time)

        self.epoch = 0
        self.callbacks = []


        #Hyperparameter stuff
        if 'hparams' in self.params.keys():
            self.hyperparameters = self.params['hparams'].copy()
        else: #infer them
            self.hyperparameters = {}

        #add also modelname as it is somehow not downloadable by tensorflow
        self.hyperparameters['MODELNAME'] = self.model_name
        self.hyperparameters['FRAMEWORK'] = config.DL_FRAMEWORK
        self.hyperparameters['EXPERIMENT'] = self.train_start_time
        self.hyperparameters['BATCH_SIZE'] = config.BATCH_SIZE
        self.hyperparameters['N_EPOCHS'] = config.EPOCHS
        self.hyperparameters = {**self.hyperparameters, **self.params['data']}
        self.hyperparameters['EARLYSTOP_METRIC'] = config.EARLY_STOP_METRIC
        self.hyperparameters['EARLYSTOP_PATIENCE'] = config.EARLY_STOP_PATIENCE

        self.hyperparameters['EARLYSTOP_PATIENCE'] = config.EARLY_STOP_PATIENCE
        #get all callbacks
        script_methods = [method for method, _ in inspect.getmembers(callbacks, inspect.isfunction)]
        for method in script_methods:
            self.hyperparameters[method] = False




            #Log the hyperparameters and training metrics
        self.metric_dict = get_metric_dict()
        log_hparams_metrics(self.writer,
                            hparam_dict=self.hyperparameters,
                            metric_dict=self.metric_dict,
                            epoch = self.epoch
                            )


    def train(self, n_epochs=config.EPOCHS):
        """
        Trains the model for a specified number of epochs.

        This method manages the main training loop of the model. For each epoch, it performs several steps. It first
        puts the model into training mode and loops over the training dataset, calculating the loss and accuracy
        for each batch and optimizing the model parameters. It logs these metrics and updates a progress bar. At
        the end of each epoch, it evaluates the model on the validation set and checks whether early stopping
        criteria have been met. If the early stopping metric has improved, it saves the current model and its parameters.
        If not, it increments a counter and potentially stops training if the counter exceeds the allowed patience.
        Finally, it steps the learning rate scheduler and calls any registered callbacks.

        #. The method first puts the model into training mode and initializes some lists and counters.
        #. Then it enters the main loop over the training data, updating the model and logging metrics.
        #. It evaluates the model on the validation set and checks the early stopping criteria.
        #. If the criteria are met, it saves the model and its parameters; if not, it increments a patience counter.
        #. It steps the learning rate scheduler and calls any callbacks.

        Args:
            n_epochs (int): The number of epochs for which the model should be trained.

        Functionality:
            This method coordinates the training of the model over a series of epochs, handling batch-wise loss
            computation, backpropagation, optimization, validation, early stopping, and model checkpoint saving.

        :param n_epochs: Number of epochs for training.
        :type n_epochs: int

        :returns: None
        :rtype: None

        .. note::
            This method modifies the state of the model and its optimizer, as well as various attributes of the Trainer instance itself.

        .. warning::
            If you set the patience value too low in the constructor, the model might stop training prematurely.
        """

        warnings.warn(f"Warning! Will only Train/Validate/Test for {config.LIMIT_EPOCHS} and {config.LIMIT_BATCHES} batches,"
                      f"as FAST_DEV_RUN is set to {config.FAST_DEV_RUN}")
        phase = "Train"
        for epoch in range(n_epochs if not config.FAST_DEV_RUN else config.LIMIT_EPOCHS):
            print(f"Epoch {epoch + 1}/{n_epochs if not config.FAST_DEV_RUN else config.LIMIT_EPOCHS}", flush=True)
            time.sleep(0.25)  # time to flush std out

            train_losses = []
            train_accuracies = []
            phase_metrics = dict({x:[] for x in config.LOG_METRICS})

            self.model.train_mode()

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training progress")

            total_loss = 0
            total_acc = 0
            preds_list,targets_list = [],[]


            print(end='', flush=True)
            for i, batch in pbar:
                if config.FAST_DEV_RUN & (i > config.LIMIT_BATCHES):
                    break
                loss, acc, labels, preds = self.model.training_step(batch)

                preds_list.append(preds)
                targets_list.append(labels)

                self.model.optimize()

                if config.DL_FRAMEWORK == "tensorflow":
                    total_loss += loss.numpy()
                    total_acc += acc.numpy()
                else:
                    total_loss += loss
                    total_acc += acc

                pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

                #append all the predictions to a list....
                train_losses.append(loss)
                train_accuracies.append(acc)

                #calculate metrics and append to phase_metrics
                # self.calculate_metrics(acc, labels, loss, phase_metrics, preds)


            print(end='', flush=True)

            self.metric_dict[f'{"Loss"}/{phase}'] = total_loss / (i + 1)
            self.calc_metric(phase, preds_list, targets_list)
            #
            #logging is done in the validation....
            # log_hparams_metrics(self.writer,
            #                     hparam_dict=self.hyperparameters,
            #                     metric_dict=self.metric_dict,
            #                     epoch=self.epoch)
            print(end='', flush=True)
            #
            # avg_train_loss = np.mean(train_losses)
            # avg_train_acc = np.mean(train_accuracies)

            self.epoch = epoch

            #Validation
            val_metrics = self.evaluate()

            metric = val_metrics[f'{Metric[config.EARLY_STOP_METRIC.capitalize()].value}/Validation']

            # check if early_stop_criterion has improved
            if config.EARLY_STOP_MODE == "min":
                early_stop_criterion = metric - config.EARLY_STOP_TOLERENCE < self.best_val_metric
            else:
                early_stop_criterion = metric + config.EARLY_STOP_TOLERENCE > self.best_val_metric

            # Check for early stopping
            if early_stop_criterion:
                self.best_val_metric = metric
                self.patience_counter = 0

                # Save the model checkpoint when Early-Stop-Metric loss improves

                os.makedirs(self.checkpoint_path, exist_ok=True)
                checkpoint_filepath = os.path.join(self.checkpoint_path, f"{self.model_name}_best_model.ckpt")
                self.model.save_checkpoint(checkpoint_filepath)

                # Save model params
                checkpoint_param_path = os.path.join(self.checkpoint_path, f"{self.model_name}_best_model_params.yaml")
                with open(checkpoint_param_path, 'w') as outfile:
                    yaml.dump(self.params, outfile, default_flow_style=False)

                #save the parameters
                with open(os.path.join(self.checkpoint_path,'config.yaml'), 'w') as file:
                    yaml.dump(self.config_dict, file)

                print(f"Best model and parameters saved at epoch {epoch + 1}")


            else:
                self.patience_counter += 1
                print(f'No improvement in {config.EARLY_STOP_METRIC} for {self.patience_counter} epoch(s)')

            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            self.model.step_scheduler()
            print("")

            for callback in self.callbacks:
                callback(self)

    def calc_metric(self, phase, preds_list, targets_list):
        # calculate average metrics for the phase
        if self.DL_FRAMEWORK == "pytorch":
            allpreds = torch.concatenate(preds_list)
            alltargets = torch.concatenate(targets_list)
        else:
            allpreds = np.concatenate(preds_list)
            alltargets = np.concatenate(targets_list)
        for log_metric in config.LOG_METRICS:
            if log_metric == "F1Score":
                self.metric_dict[f'{log_metric}/{phase}'] = self.model.calculate_f1score(allpreds, alltargets)
            elif log_metric == "Accuracy":
                self.metric_dict[f'{log_metric}/{phase}'] = self.model.calculate_accuracy(allpreds, alltargets)
            elif log_metric == "Recall":
                self.metric_dict[f'{log_metric}/{phase}'] = self.model.calculate_recall(allpreds, alltargets)
            elif log_metric == "Precision":
                self.metric_dict[f'{log_metric}/{phase}'] = self.model.calculate_precision(allpreds, alltargets)

            # self.metric_dict[f'{log_metric}/{phase}'] = np.array(phase_metrics[log_metric]).mean()

    def calculate_metrics(self, acc, labels, loss, phase_metrics, preds):
        for log_metric in config.LOG_METRICS:
            if log_metric.lower() == "precision":
                phase_metrics[log_metric].append(self.model.calculate_precision(preds, labels))
            elif log_metric.lower() == "recall":
                phase_metrics[log_metric].append(self.model.calculate_recall(preds, labels))
            elif log_metric.lower() == "f1score":
                phase_metrics[log_metric].append(self.model.calculate_f1score(preds, labels))
            elif log_metric.lower() == "accuracy":
                phase_metrics[log_metric].append(acc)
            elif log_metric.lower() == "loss":
                phase_metrics[log_metric].append(loss)


    def evaluate(self):
        """
        Evaluates the model on the validation set.

        This method sets the model to evaluation mode and loops over the validation dataset, computing the loss and
        accuracy for each batch. It then averages these metrics and logs them. This process provides an unbiased
        estimate of the model's performance on new data during training.

        Functionality:
            It manages the evaluation of the model on the validation set, handling batch-wise loss computation and
            accuracy assessment.

        :returns: Average validation loss and accuracy
        :rtype: Tuple[float, float]

        .. warning::
            Ensure the model is in evaluation mode to correctly compute the validation metrics.
        """

        self.model.eval_mode()

        valid_losses = []
        valid_accuracies = []
        preds_list, targets_list = [], []
        phase_metrics = dict({x: [] for x in config.LOG_METRICS})

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc=f"Validation progress")

        total_loss = 0
        total_acc = 0

        phase = "Validation"
        print(end='', flush=True)
        for i, batch in pbar:
            loss, acc, labels, preds = self.model.validation_step(batch)

            if config.DL_FRAMEWORK == "tensorflow":
                total_loss += loss.numpy()
                total_acc += acc.numpy()
            else:
                total_loss += loss
                total_acc += acc

            valid_losses.append(loss)
            valid_accuracies.append(acc)
            preds_list.append(preds)
            targets_list.append(labels)

            # #calculate metrics and append to phase_metrics
            # self.calculate_metrics(acc, labels, loss, phase_metrics, preds)

            pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})
            print(end='', flush=True)

        print(end='', flush=True)

        #
        #append loss
        self.metric_dict[f'{"Loss"}/{phase}'] = total_loss / (i + 1)
        self.calc_metric(phase, preds_list, targets_list)

        # calculate average metrics for the phase
        # for log_metric in config.LOG_METRICS:
        #     self.metric_dict[f'{log_metric}/{phase}'] = np.array(phase_metrics[log_metric]).mean()
        self.metric_dict['LearningRate'] = self.model.get_lr()

        # log the metrics to the dict
        log_hparams_metrics(self.writer,
                            hparam_dict=self.hyperparameters,
                            metric_dict=self.metric_dict,
                            epoch=self.epoch)


        print(flush=True)

        return self.metric_dict

    def test(self):
        """
        Tests the model on the test set.

        This method loads the best saved model, sets it to evaluation mode, and then loops over the test dataset,
        computing the loss, accuracy, and predictions for each batch. It then averages the loss and accuracy and logs
        them. It also collects all the model's predictions and their corresponding labels.

        Functionality:
            It manages the testing of the model on the test set, handling batch-wise loss computation, accuracy
            assessment, and prediction generation.

        :returns: List of all predictions and their corresponding labels
        :rtype: Tuple[List, List]

        """
        checkpoint_filepath = os.path.join(self.checkpoint_path, f"{self.model_name}_best_model.ckpt")
        print(f"\nRetrieving best model saved at {checkpoint_filepath}")
        self.model.load_checkpoint(checkpoint_filepath)

        self.model.eval_mode()

        test_losses = []
        test_accuracies = []
        all_preds,all_labels = [],[]
        total_loss = 0
        total_acc = 0

        phase_metrics = dict({x: [] for x in config.LOG_METRICS})
        phase = "Test"
        for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader),
                             desc=f"Testing progress"):
            loss, acc, labels, preds = self.model.test_step(batch)

            if config.DL_FRAMEWORK == "tensorflow":
                total_loss += loss.numpy()
                total_acc += acc.numpy()
            else:
                total_loss += loss
                total_acc += acc

            test_losses.append(loss)
            test_accuracies.append(acc)
            all_preds.append(preds)
            all_labels.append(batch[1])
            #calculate metrics and append to phase_metrics
            # self.calculate_metrics(acc, labels, loss, phase_metrics, preds)


        # calculate average metrics for the phase
        # for log_metric in config.LOG_METRICS:
        #     self.metric_dict[f'{log_metric}/{phase}'] = np.array(phase_metrics[log_metric]).mean()
        self.metric_dict[f'{"Loss"}/{phase}'] = total_loss / (i + 1)
        self.calc_metric(phase, all_preds, all_labels)

        # log the metrics to the dict
        log_hparams_metrics(self.writer,
                            hparam_dict=self.hyperparameters,
                            metric_dict=self.metric_dict,
                            epoch=self.epoch)

        print(flush=True)


        print(f"Test/Accuracy: {self.metric_dict[f'Accuracy/{phase}']}")
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_labels)

        self.writer.close()
        return preds, targets


    def write_classification_report(self,preds,labels):
        """
        Function to export classification report

        Args:


        :param preds: tensor
        :type preds: tensor
        :param labels: tensor
        :return: None
        """
        report = self.get_classification_report(labels, preds)
        with open(os.path.join(self.log_dir,"TEST_classificationReport.txt"), "w") as infile:
            infile.write(report)



    def get_classification_report(self, labels, preds,verbose = True):
        """
        Generates Classification report
        :param labels: tensor with True values
        :param preds:  tensor with predicted values
        :param verbose: Wherer to print the report
        :type: verbose: bool
        :return: classification report
        """
        report = classification_report(labels, np.argmax(preds, axis=1),
                              labels=list(self.dataset.target_dict.keys()),
                              target_names=list(self.dataset.target_dict.values()))
        if verbose:
            print(report)
        return report

    def add_callback(self, callback):
        """
        Adds a callback to the Trainer.

        This method simply appends a callback function to the list of callbacks stored by the Trainer instance.
        These callbacks are called at the end of each training epoch.

        Functionality:
            It allows the addition of custom callbacks to the training process, enhancing its flexibility.

        :param callback: The callback function to be added.
        :type callback: Callable

        :returns: None
        :rtype: None

        .. warning::
            The callback function must be callable and should not modify the training process.
        """
        #Add the setting to the hyperparameters
        self.hyperparameters[callback.__name__] = True
        self.callbacks.append(callback)


if __name__ == '__main__':
    # Get Data
    trainer = Trainer(
                      config=config,)
    trainer.add_callback(dropout_callback)
    trainer.add_callback(augmentation_increase_callback)
    trainer.train()
    preds,labels = trainer.test()
    trainer.write_classification_report(preds,labels)
