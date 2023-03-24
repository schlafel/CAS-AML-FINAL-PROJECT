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
from sklearn import *
from torchmetrics.classification import accuracy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import *

from data_models import ASL_DATSET, ASLDataModule
from models import LSTM_BASELINE_Model, LSTM_Predictor

if __name__ == '__main__':
    """
    #1. Load data.
    
    #2. Create Model
    
    #3. Setting up Callbacks & Trainer
    
    #4. Fit the Model
    """


    MAX_SEQUENCES = 150
    BATCH_SIZE = 512
    num_workers = os.cpu_count()#or 0


    # ------------ 1. Load data ------------
    dM = ASLDataModule(batch_size=BATCH_SIZE,
                       max_seq_length=MAX_SEQUENCES,
                       num_workers= num_workers)
    dM.setup()
    t_dl = dM.train_dataloader()

    # for sample in t_dl:
    #     pass


    # ------------ 2. Create Model PL------------
    model = LSTM_Predictor(n_features=188,
                           num_layers=3)

    # ------------ 3. Create Model Callbacks------------
    #Model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ROOT_PATH, "checkpoints"),
        filename="best_checkpoint",
        save_top_k=1,
        monitor="train_loss",
        verbose=True,
        mode="min"
    )
    #Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=os.path.join(ROOT_PATH, "checkpoints"),
                                  name="lightning_logs",
                                  version="FIRST_POC_MODEL"
                                  )

    trainer = pl.Trainer(accelerator="gpu",
                         logger=tb_logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=250,
                         )


    # ------------ 3. Create Model Callbacks------------
    trainer.fit(model=model,
                datamodule=dM
                )





