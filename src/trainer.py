import sys
sys.path.insert(0, '../src')

from config import *
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader

    def train(self, n_epochs=EPOCHS):
        for epoch in range(n_epochs):
            train_losses = []
            train_accuracies = []

            self.model.train_mode()

            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                 desc=f"Training progress"):

                loss, acc = self.model.training_step(batch)
                self.model.optimize()

                train_losses.append(loss.item())
                train_accuracies.append(acc)

            self.model.step_scheduler()

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            print(f"EPOCH {epoch+1:>3}: Train accuracy: {avg_train_acc:>3.2f}, Train Loss: {avg_train_loss:>9.8f}")

            val_loss, val_acc = self.evaluate()
            print(f"EPOCH {epoch+1:>3}: Validation accuracy: {val_acc:>3.2f}, Validation Loss: {val_loss:>9.8f}\n")

    def evaluate(self):
        self.model.eval_mode()

        valid_losses = []
        valid_accuracies = []

        for i, batch in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                             desc=f"Validation progress"):
            loss, acc = self.model.validation_step(batch)

            valid_losses.append(loss.cpu())
            valid_accuracies.append(acc)

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

            test_losses.append(loss.cpu())
            test_accuracies.append(acc.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch[1])

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)

        print(f"Test Accuracy: {avg_test_acc:>3.2f}, Test Loss: {avg_test_loss:>9.8f}")

        return all_preds, all_labels

import importlib
from data.dataset import ASL_DATASET
from data.data_utils import create_data_loaders
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor, TQDMProgressBar,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

if __name__ == '__main__':
    print(DL_FRAMEWORK)

    #module_name = f"models.{DL_FRAMEWORK}.lightning_models"
    #class_name = "LightningTransformerPredictor"

    module_name = f"models.{DL_FRAMEWORK}.models"
    class_name = "TransformerPredictor"

    module = importlib.import_module(module_name)
    TransformerPredictorModel = getattr(module, class_name)

    print(TransformerPredictorModel)

    asl_dataset = ASL_DATASET(augment=True, augmentation_threshold=0.3)
    train_ds, val_ds, test_ds = create_data_loaders(asl_dataset,
                                                               batch_size=BATCH_SIZE,
                                                               dl_framework=DL_FRAMEWORK,
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
    model = model.float().to(DEVICE)
    print(model)

    trainer = Trainer(model, train_ds, val_ds, test_ds)

    trainer.train()

    checkpoint_callback = ModelCheckpoint(
        filename=class_name + "-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        monitor="val_accuracy",
        verbose=True,
        mode="max"
    )

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(ROOT_PATH, "lightning_logs"),
        name=class_name,
        # version=mod_name
    )

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.005,
        patience=6,
        verbose=True,
        mode='max'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        enable_progress_bar=True,
        accelerator="gpu",
        logger=tb_logger,
        callbacks=[
            DeviceStatsMonitor(),
            early_stop_callback,
            checkpoint_callback,
            MyProgressBar(),
            lr_monitor
        ],
        max_epochs=100,
        # limit_train_batches=10,
        # limit_val_batches=0,
        num_sanity_val_steps=0,
        profiler=None,  # select from None
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_ds,
        val_dataloaders=val_ds,

    )

