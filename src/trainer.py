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
import sys

sys.path.insert(0, '../src')

from config import *
import numpy as np
from tqdm import tqdm

import time

import importlib
from torch.utils.tensorboard import SummaryWriter
from dl_utils import get_model_params, log_metrics
from data.data_utils import create_data_loaders
from data.dataset import ASL_DATASET
from datetime import datetime
from callbacks import dropout_callback, augmentation_increase_callback
import yaml

class Trainer:
    """
    A trainer class which acts as a control hub for the model lifecycle, including initial setup, executing training
    epochs, performing validation and testing, implementing early stopping, and logging results. The module has been
    designed to be agnostic to the specific deep learning framework, enhancing its versatility across various projects.
    """
    def __init__(self, modelname=MODELNAME, dataset=ASL_DATASET, patience=EARLY_STOP_PATIENCE,
                 enableAugmentationDropout=True, augmentation_threshold=0.35):
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
            modelname (str): The name of the model to be used for training.
            dataset (Dataset): The dataset to be used.
            patience (int): The number of epochs with no improvement after which training will be stopped.
            enableAugmentationDropout (bool): If True, enable data augmentation dropout.
            augmentation_threshold (float): The threshold for data augmentation.

        Functionality:
            This method initializes various components, such as the model, dataset, data loaders,
            logging writer, and checkpoint path, required for the training process.

        :param modelname: The name of the model for training.
        :type modelname: str
        :param dataset: The dataset for training.
        :type dataset: Dataset
        :param patience: The number of epochs with no improvement after which training will be stopped.
        :type patience: int
        :param enableAugmentationDropout: If True, enable data augmentation dropout.
        :type enableAugmentationDropout: bool
        :param augmentation_threshold: The threshold for data augmentation.
        :type augmentation_threshold: float

        :rtype: None

        .. note::
            This method only initializes the Trainer class. The actual training is done by calling the train() method.

        .. warning::
            Make sure the specified model name corresponds to an actual model in your project's models directory.
        """
        self.model_name = modelname
        module_name = f"models.{DL_FRAMEWORK}.models"
        self.params = get_model_params(modelname)

        module = importlib.import_module(module_name)
        Model = getattr(module, modelname)

        # Get Model
        self.model = Model(**self.params)
        print(f"Using model: {module_name}.{modelname}")

        # Get Data
        asl_dataset = dataset(augment=True, augmentation_threshold=augmentation_threshold, enableDropout=enableAugmentationDropout)
        self.train_loader, self.valid_loader, self.test_loader = create_data_loaders(
            asl_dataset, batch_size=BATCH_SIZE, dl_framework=DL_FRAMEWORK, num_workers=4)

        batch = next(iter(self.train_loader))[0]
        self.model(batch)

        self.patience = patience
        self.best_val_metric = float('inf') if EARLY_STOP_MODE == "min" else float('-inf')
        self.patience_counter = 0

        now = datetime.now()

        self.model_class = self.model.__class__.__name__
        self.train_start_time = now.strftime("%Y-%m-%d %H_%M")

        self.writer = SummaryWriter(os.path.join(ROOT_PATH, RUNS_DIR, DL_FRAMEWORK,
                                                 self.model_class, self.train_start_time))

        self.checkpoint_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, self.model_class, self.train_start_time)

        self.epoch = 0
        self.callbacks = []

    def train(self, n_epochs=EPOCHS):
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
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}", flush=True)
            time.sleep(0.25)  # time to flush std out

            train_losses = []
            train_accuracies = []

            self.model.train_mode()

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training progress")

            total_loss = 0
            total_acc = 0

            print(end='', flush=True)
            for i, batch in pbar:
                loss, acc = self.model.training_step(batch)
                self.model.optimize()

                total_loss += loss
                total_acc += acc

                pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

                train_losses.append(loss)
                train_accuracies.append(acc)

            print(end='', flush=True)

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            log_metrics('Train', avg_train_loss, avg_train_acc, epoch, self.model.get_lr(), self.writer)
            print(end='', flush=True)

            self.epoch = epoch

            val_loss, val_acc = self.evaluate()

            if EARLY_STOP_METRIC == "loss":
                metric = val_loss
            elif EARLY_STOP_METRIC == "accuracy":
                metric = val_acc

            # check if early_stop_criterion has improved
            if EARLY_STOP_MODE == "min":
                early_stop_criterion = metric - EARLY_STOP_TOLERENCE < self.best_val_metric
            else:
                early_stop_criterion = metric + EARLY_STOP_TOLERENCE > self.best_val_metric

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
                print(f"Best model and parameters saved at epoch {epoch + 1}")

            else:
                self.patience_counter += 1
                print(f'No improvement in {EARLY_STOP_METRIC} for {self.patience_counter} epoch(s)')

            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            self.model.step_scheduler()
            print("")

            for callback in self.callbacks:
                callback(self)


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

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc=f"Validation progress")

        total_loss = 0
        total_acc = 0

        print(end='', flush=True)
        for i, batch in pbar:
            loss, acc = self.model.validation_step(batch)

            valid_losses.append(loss)
            valid_accuracies.append(acc)

            total_loss += loss
            total_acc += acc

            pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

        print(end='', flush=True)
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)

        log_metrics('Validation', avg_valid_loss, avg_valid_acc, self.epoch, self.model.get_lr(), self.writer)
        print(flush=True)

        return avg_valid_loss, avg_valid_acc

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
        all_preds = []
        all_labels = []
        for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader),
                             desc=f"Testing progress"):
            loss, acc, preds = self.model.test_step(batch)

            test_losses.append(loss)
            test_accuracies.append(acc)
            all_preds.append(preds)
            all_labels.append(batch[1])

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)

        log_metrics('Test', avg_test_loss, avg_test_acc, self.epoch, self.model.get_lr(), self.writer)
        print(flush=True)
        self.writer.close()

        return all_preds, all_labels

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
        self.callbacks.append(callback)


if __name__ == '__main__':
    # Get Data
    trainer = Trainer(modelname=MODELNAME,
                      enableAugmentationDropout=False,
                      augmentation_threshold=0.35)
    trainer.add_callback(dropout_callback)
    trainer.add_callback(augmentation_increase_callback)
    trainer.train()
    trainer.test()

