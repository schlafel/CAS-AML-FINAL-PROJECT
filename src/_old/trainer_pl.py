# main.py
from pytorch_lightning.cli import LightningCLI,SaveConfigCallback
import src
from config import *
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.extend(['C:\\Users\\fs.GUNDP\\Python\\CAS-AML-FINAL-PROJECT', 'C:\\Users\\fs.GUNDP\\Python\\CAS-AML-FINAL-PROJECT', 'C:/Users/fs.GUNDP/Python/CAS-AML-FINAL-PROJECT'])

# simple demo classes for your convenience
from models.models import TransformerPredictor
from data.data_utils import ASL_DATAMODULE

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, TQDMProgressBar,EarlyStopping

def cli_main():

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(ROOT_PATH, "lightning_logs"),
        name="test",
        # version=mod_name
    )


    cli = LightningCLI(TransformerPredictor, ASL_DATAMODULE,
                       save_config_callback = SaveConfigCallback,
                       trainer_defaults=dict(
                           accelerator = "gpu",
                           # logger = TensorBoardLogger,
                       ),

                       parser_kwargs={"parser_mode": "omegaconf"},
                       save_config_kwargs={"overwrite": True},
                       # run = True
                       )
    
    # cli.trainer.fit(cli.model)

    # cli.trainer.test(ckpt_path = "best")
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block