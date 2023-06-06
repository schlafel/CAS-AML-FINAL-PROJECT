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

import mlflow

from ray import air, tune
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune.schedulers import ASHAScheduler


def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


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


def train_function(config):
    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back to Tune.
        session.report({"iterations": step,
                        "mean_loss": intermediate_score})
        time.sleep(0.1)


def tune_with_callback(mlflow_tracking_uri, finish_fast=False):
    search_space = {
        "learning_rate": tune.loguniform(0.0001, 0.001),
        "dropout": tune.uniform(0.1, 0.4),
        "num_layers": tune.choice([2, 3, 4, 5,6]),
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


def train_function_mlflow(config):
    setup_mlflow(config)

    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Log the metrics to mlflow
        mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
        # Feed the score back to Tune.
        session.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(0.1)


def tune_with_setup(mlflow_tracking_uri, finish_fast=False):
    # Set the experiment, or create a new one if does not exist yet.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name="mixin_example")
    tuner = tune.Tuner(
        train_function_mlflow,
        run_config=air.RunConfig(
            name="mlflow",
        ),
        tune_config=tune.TuneConfig(
            num_samples=5,
        ),
        param_space={
            "width": tune.randint(10, 100),
            "height": tune.randint(0, 100),
            "steps": 5 if finish_fast else 100,
            "mlflow": {
                "experiment_name": "mixin_example",
                "tracking_uri": mlflow.get_tracking_uri(),
            },
        },
    )
    tuner.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        help="The tracking URI for the MLflow tracking server.",
    )
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        mlflow_tracking_uri = os.path.join(tempfile.gettempdir(), "mlruns")
    else:
        mlflow_tracking_uri = args.tracking_uri

    tune_with_callback(mlflow_tracking_uri, finish_fast=args.smoke_test)

