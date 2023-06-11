import pytorch_lightning as pl
import sys
# from sklearn import *

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor, TQDMProgressBar,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import *
from src.data import data_utils
from src.data.dataset import ASL_DATASET

from src._old.models import TransformerPredictor


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
    """
    #1. Load data.
    
    #2. Create Model
    
    #3. Setting up Callbacks & Trainer
    
    #4. Fit the Model
    """

    # MAX_SEQUENCES = 150
    BATCH_SIZE = 256  # Not optimal as not a perfect power of 2, but maximum that fits in my GPU
    num_workers = 4#os.cpu_count() // 2  # or 0
    mod_name = "FIRST_TRANSFORMER_MODEL_2"
    DL_FRAMEWORK = "PYTORCH"

    # ------------ 1. Load data ------------
    asl_dataset = ASL_DATASET(augment=True,
                              augmentation_threshold=0.3)
    # train_ds,val_ds,test_ds = get_dataloader(asl_dataset,shuffle=True,dl_framework="PYTORCH")
    train_ds, val_ds, test_ds = data_utils.create_data_loaders(asl_dataset,
                                                               batch_size=BATCH_SIZE,
                                                               dl_framework=DL_FRAMEWORK,
                                                               num_workers=num_workers)


    print(f'Got the lengths for Train-Dataset: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}')

    # dM.setup()
    # dL = dM.train_dataloader()
    print(next(iter(train_ds))[0].shape)
    batch = next(iter(train_ds))[0]
    # ------------ 2. Create Model PL------------
    model = TransformerPredictor(
        d_model=192,
        n_head=8,
        dim_feedforward=512,
        dropout=0.3,
        layer_norm_eps=1e-6,
        norm_first=False,
        batch_first=True,
        num_layers=3,
        num_classes=250,
        learning_rate = LEARNING_RATE
    )
    model(batch)
    print(model)

    # ------------ 3. Create Model Callbacks------------
    # Model checkpoints
    checkpoint_callback = ModelCheckpoint(
        filename=mod_name + "-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        monitor="val_accuracy",
        verbose=True,
        mode="max"
    )
    # Tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(ROOT_PATH, "lightning_logs"),
        name=mod_name,
        # version=mod_name
    )

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.005,
        patience=6,
        verbose=True,
        mode='max'
    )

    #lr_monitor callback
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

    # ------------ 4. Tune Model ------------
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(
    #     model=model,
    #     datamodule=dM,
    #     init_val = 256
    # )
    # 1024 is optimal...

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