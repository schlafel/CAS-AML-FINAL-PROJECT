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
    def __init__(self, modelname=MODELNAME, dataset=ASL_DATASET, patience=EARLY_STOP_PATIENCE,
                 enableAugmentationDropout=True, augmentation_threshold=0.35):

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
                print(f'No improvement in loss for {self.patience_counter} epoch(s)')

            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            self.model.step_scheduler()
            print("")

            for callback in self.callbacks:
                callback(self)


    def evaluate(self):
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
        self.callbacks.append(callback)


if __name__ == '__main__':
    # Get Data
    trainer = Trainer(modelname='YetAnotherEnsemble', enableAugmentationDropout=False, augmentation_threshold=0.05)
    trainer.add_callback(dropout_callback)
    trainer.add_callback(augmentation_increase_callback)
    trainer.train()
    trainer.test()

