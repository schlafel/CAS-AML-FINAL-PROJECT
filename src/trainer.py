import sys
sys.path.insert(0, '../src')

from config import *
import numpy as np
from tqdm import tqdm

import time

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader

    def train(self, n_epochs=EPOCHS):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}", flush=True)
            time.sleep(0.5)  # time to flush std out

            train_losses = []
            train_accuracies = []

            self.model.train_mode()

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                 desc=f"Training progress")

            total_loss = 0
            total_acc = 0

            for i, batch in pbar:

                loss, acc = self.model.training_step(batch)
                self.model.optimize()

                total_loss += loss
                total_acc += acc

                pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

                train_losses.append(loss)
                train_accuracies.append(acc)

            self.model.step_scheduler()

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)

            print(f"EPOCH {epoch+1:>3}: Train accuracy: {avg_train_acc:>3.2f}, Train Loss: {avg_train_loss:>9.8f}",
                  flush=True)

            val_loss, val_acc = self.evaluate()
            print(f"EPOCH {epoch+1:>3}: Validation accuracy: {val_acc:>3.2f}, Validation Loss: {val_loss:>9.8f}",
                  flush=True)

            print(flush=True)

    def evaluate(self):
        self.model.eval_mode()

        valid_losses = []
        valid_accuracies = []

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc=f"Validation progress")

        total_loss = 0
        total_acc = 0

        for i, batch in pbar:
            loss, acc = self.model.validation_step(batch)

            valid_losses.append(loss)
            valid_accuracies.append(acc)

            total_loss += loss
            total_acc += acc

            pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)

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

        print(f"Test Accuracy: {avg_test_acc:>3.2f}, Test Loss: {avg_test_loss:>9.8f}")

        return all_preds, all_labels

import importlib
from data.dataset import ASL_DATASET
from data.data_utils import create_data_loaders


if __name__ == '__main__':
    DL_FRAMEWORK='tensorflow'

    module_name = f"models.{DL_FRAMEWORK}.models"
    class_name = "TransformerPredictor"

    print(f"Using model: {module_name}.{class_name}")

    module = importlib.import_module(module_name)
    TransformerPredictorModel = getattr(module, class_name)

    asl_dataset = ASL_DATASET(augment=True, augmentation_threshold=0.3)
    train_ds, val_ds, test_ds = create_data_loaders(asl_dataset,batch_size=BATCH_SIZE,dl_framework=DL_FRAMEWORK,
                                                               num_workers=4)

    batch = next(iter(train_ds))[0]

    params = {
        "d_model": 192,
        "n_head": 8,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "norm_first": True,
        "batch_first": True,
        "num_layers": 2,
        "num_classes": 250,
        "learning_rate": 0.001
    }

    model = TransformerPredictorModel(**params)

    model(batch)
    if DL_FRAMEWORK=='pytorch':
        model = model.float().to(DEVICE)

    trainer = Trainer(model, train_ds, val_ds, test_ds)

    trainer.train()


