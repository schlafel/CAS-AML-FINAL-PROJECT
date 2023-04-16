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

from data_models import ASL_DATSET, ASLDataModule, ASLDataModule_Preprocessed
from src.models.models import LSTM_BASELINE_Model, LSTM_Predictor, TransformerPredictor

if __name__ == '__main__':
    """
    #1. Load data.
    
    #2. Create Model
    
    #3. Setting up Callbacks & Trainer
    
    #4. Fit the Model
    """

    MAX_SEQUENCES = 150
    BATCH_SIZE = 64  #Not optimal as not a perfect power of 2, but maximum that fits in my GPU
    num_workers = os.cpu_count() // 2  # or 0
    mod_name = "FIRST_TRANSFORMER_MODEL_2"

    # ------------ 1. Load data ------------
    dM = ASLDataModule_Preprocessed(batch_size=BATCH_SIZE,
                                    max_seq_length=MAX_SEQUENCES,
                                    num_workers=num_workers)
    dM.setup()
    dL = dM.train_dataloader()
    print(next(iter(dL))[0].shape)
    batch = next(iter(dL))[0]
    # ------------ 2. Create Model PL------------
    model = TransformerPredictor(seq_length=MAX_SEQUENCES,
                                 hidden_size=436,
                                 num_heads=2
                                 )

    print(model)
    model(batch)
    # ------------ 3. Create Model Callbacks------------
    # Model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ROOT_PATH, "checkpoints"),
        filename=mod_name + "-{epoch:02d}-{train_accuracy:.2f}",
        save_top_k=1,
        monitor="train_loss",
        verbose=True,
        mode="min"
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
        max_epochs=50,
        limit_val_batches=0,
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
        datamodule=dM,
    )
