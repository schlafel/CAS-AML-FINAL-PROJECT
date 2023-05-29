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

class Trainer:
    def __init__(self, modelname=MODELNAME, dataset=ASL_DATASET, patience=EARLY_STOP_PATIENCE):

        self.model_name = modelname
        module_name = f"models.{DL_FRAMEWORK}.models"
        params = get_model_params(modelname)

        module = importlib.import_module(module_name)
        TransformerPredictorModel = getattr(module, modelname)

        # Get Model
        self.model = TransformerPredictorModel(**params)
        print(f"Using model: {module_name}.{modelname}")

        asl_dataset = dataset(augment=True, augmentation_threshold=0.3)
        self.train_loader, self.valid_loader, self.test_loader = create_data_loaders(
            asl_dataset, batch_size=BATCH_SIZE, dl_framework=DL_FRAMEWORK, num_workers=4)

        self.model(next(iter(self.train_loader))[0])

        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        now = datetime.now()

        self.model_class = self.model.__class__.__name__
        self.writer = SummaryWriter(os.path.join(ROOT_PATH, RUNS_DIR, DL_FRAMEWORK,
                                                 self.model_class, now.strftime("%Y-%m-%d %H:%M")))

        self.checkpoint_path = os.path.join(ROOT_PATH, CHECKPOINT_DIR, DL_FRAMEWORK, self.model_class)

        self.epoch = 0

    def train(self, n_epochs=EPOCHS):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}", flush=True)
            time.sleep(0.5)  # time to flush std out

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

            # Check for early stopping
            if val_loss - EARLY_STOP_TOLERENCE < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save the model checkpoint when validation loss improves

                os.makedirs(self.checkpoint_path, exist_ok=True)
                checkpoint_filepath = os.path.join(self.checkpoint_path, f"{self.model_name}_best_model.ckpt")
                self.model.save_checkpoint(checkpoint_filepath)
                print(f"Best model saved at epoch {epoch + 1}")

            else:
                self.patience_counter += 1
                print(f'No improvement in loss for {self.patience_counter} epoch(s)')

            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            self.model.step_scheduler()
            print("")

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


if __name__ == '__main__':
    # Get Data
    trainer = Trainer(modelname='TransformerEnsemble')
    trainer.train()
    trainer.test()
