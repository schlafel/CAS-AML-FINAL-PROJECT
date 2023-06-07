#!/usr/bin/env python
"""Examples using MLfowLoggerCallback and setup_mlflow.
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
import yaml
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
import mlflow

import os
import tempfile
import time

from trainer import Trainer

import mlflow

from ray import air, tune
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune.schedulers import ASHAScheduler


class Trainer_HparamSearch(Trainer):
    def __init__(self,  modelname=MODELNAME, dataset=ASL_DATASET, patience=EARLY_STOP_PATIENCE):
        super.__init__( modelname, dataset, patience)



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


def train(config):
    # mlflow.autolog()
    # with mlflow.start_run():

    module_name = f"models.{DL_FRAMEWORK}.models"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, MODELNAME)

    base_config = get_model_params(MODELNAME)
    for key, val in config.items():
        base_config[key] = val
    # Get Model
    model = model_cls(**base_config)
    print(f"Using model: {module_name}.{MODELNAME}")
    # Get Data
    asl_dataset = ASL_DATASET(augment=True, augmentation_threshold=base_config["augmentation_threshold"])
    train_loader, valid_loader, test_loader = create_data_loaders(
        asl_dataset, batch_size=BATCH_SIZE, dl_framework=DL_FRAMEWORK, num_workers=4)
    model(next(iter(train_loader))[0])
    params = config

    for epoch in range(EPOCHS):
        # print(f"Epoch {epoch + 1}/{EPOCHS}", flush=True)
        time.sleep(0.5)  # time to flush std out

        train_losses = []
        train_accuracies = []

        model.train_mode()

        # pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training progress")

        total_loss = 0
        total_acc = 0

        # print(end='', flush=True)
        for i, batch in enumerate(train_loader):
            loss, acc = model.training_step(batch)
            model.optimize()

            total_loss += loss
            total_acc += acc
            # if not TUNE_HP:
            # pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

            train_losses.append(loss)
            train_accuracies.append(acc)

        # print(end='', flush=True)

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        # print(end='', flush=True)

        # Valid
        model.eval_mode()

        valid_losses = []
        valid_accuracies = []

        # pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validation progress")

        total_loss = 0
        total_acc = 0

        # print(end='', flush=True)
        for i, batch in enumerate(valid_loader):
            loss, acc = model.validation_step(batch)

            valid_losses.append(loss)
            valid_accuracies.append(acc)

            total_loss += loss
            total_acc += acc

            # pbar.set_postfix({'Loss': total_loss / (i + 1), 'Accuracy': total_acc / (i + 1)})

        # print(end='', flush=True)
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)
        # with mlflow.start_run(nested=True):

        session.report({
            "iterations": epoch,
            "val_loss": float(avg_valid_loss),
            "val_accuracy": float(avg_valid_acc),
            "train_loss": float(avg_train_loss),
            "train_accuracy": float(avg_train_acc),
        },
        )


def tune_with_callback(
        mlflow_tracking_uri,
        finish_fast=False):
    search_space = {
        "learning_rate": tune.loguniform(0.0001, 0.001),
        "dropout": tune.uniform(0.1, 0.4),
        "num_layers": tune.choice([2, 3, 4, 5, 6]),
        "n_head": tune.choice([8]),
        "dim_feedforward": tune.choice([1024, 2048]),
        "gamma": tune.loguniform(0.92, 0.96),
        "augmentation_threshold": tune.uniform(0.2, 0.55)
    }

    scheduler = ASHAScheduler(
        max_t=EPOCHS,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(trainable=train, resources={"cpu": 3,
                                                        "gpu": 0.25}),
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=f"{DL_FRAMEWORK}_{MODELNAME}",
                    save_artifact=True,
                )
            ],
        ),
        tune_config=tune.TuneConfig(
            num_samples=30,
            scheduler=scheduler,
            search_alg=OptunaSearch(),
            metric="val_accuracy",
            mode="max",
        ),
        param_space=search_space,
    )
    tuner.fit()



if __name__ == "__main__":
    tr = Trainer_HparamSearch()
    tr.train()

    mlflow_tracking_uri = os.path.join(tempfile.gettempdir(), "mlruns")


    tune_with_callback(mlflow_tracking_uri)
