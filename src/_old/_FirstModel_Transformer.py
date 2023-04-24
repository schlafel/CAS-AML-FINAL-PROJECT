import torch
import torch.nn as nn
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
# from sklearn import *
from torchmetrics.classification import accuracy

from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from src.config import *
from src.data import data_utils

from data_models import ASL_DATSET, ASLDataModule, ASLDataModule_Preprocessed
from src.models.models import LSTM_BASELINE_Model, LSTM_Predictor, TransformerPredictor

if __name__ == '__main__':
    """
    #1. Load data.
    
    #2. Create Model
    
    #3. Setting up Callbacks & Trainer
    
    #4. Fit the Model
    """

    # MAX_SEQUENCES = 150
    BATCH_SIZE = 64  #Not optimal as not a perfect power of 2, but maximum that fits in my GPU
    num_workers = os.cpu_count() // 2  # or 0
    mod_name = "FIRST_TRANSFORMER_MODEL_2"

    # ------------ 1. Load data ------------
    asl_dataset = data_utils.ASL_DATASET(augment = True, )

    train_ds, val_ds, test_ds = data_utils.create_data_loaders(asl_dataset)

    # dM.setup()
    # dL = dM.train_dataloader()
    print(next(iter(train_ds))["landmarks"].shape)
    batch = next(iter(train_ds))["landmarks"]
    # ------------ 2. Create Model PL------------
    model = TransformerPredictor(seq_length=INPUT_SIZE,
                                 hidden_size=192,
                                 num_heads=4,
                                 dropout = .2
                                 )

    print(model)
    model(batch)
    # ------------ 3. Create Model Callbacks------------
    # Model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ROOT_PATH, "checkpoints"),
        filename=mod_name + "-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        monitor="val_accuracy",
        verbose=True,
        mode="max"
    )
    # Tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(ROOT_PATH, "checkpoints","lightning_logs"),
        name=mod_name,
        # version=mod_name
    )

    trainer = pl.Trainer(
        enable_progress_bar=True,
        accelerator="gpu",
        logger=tb_logger,
        callbacks=[DeviceStatsMonitor(),checkpoint_callback],
        max_epochs=100,
        #limit_train_batches=10,
        # limit_val_batches=0,
        num_sanity_val_steps=0,
        profiler=None,#select from None
    )

    # ------------ 4. Tune Model ------------
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(
    #     model=model,
    #     datamodule=dM,
    #     init_val = 256
    # )
    #1024 is optimal...


    # ------------ 5. Train Model ------------
    trainer.fit(
        model=model,
        train_dataloaders=train_ds,
        val_dataloaders=val_ds,


    )


    # ----------- 6. Test Model ------------

    trainer.test(ckpt_path="best",
                 dataloaders=test_ds
                 )
