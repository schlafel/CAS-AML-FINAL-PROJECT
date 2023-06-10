import sys

sys.path.insert(0, '..')
from config import DYNAMIC_DROP_OUT_REDUCTION_RATE, DYNAMIC_DROP_OUT_MAX_THRESHOLD, \
    DYNAMIC_DROP_OUT_REDUCTION_INTERVAL, DYNAMIC_AUG_INC_RATE, DYNAMIC_AUG_MAX_THRESHOLD, DYNAMIC_AUG_INC_INTERVAL
from dl_utils import get_dataset
import torch.nn as nn


def dropout_callback(trainer, dropout_rate=DYNAMIC_DROP_OUT_REDUCTION_RATE,
                     max_dropout=DYNAMIC_DROP_OUT_MAX_THRESHOLD):
    step = trainer.epoch + 1
    model = trainer.model
    if step % DYNAMIC_DROP_OUT_REDUCTION_INTERVAL == 0:
        print(f"Increasing dropout rate by {dropout_rate}")
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(module.p * dropout_rate, max_dropout)


def augmentation_increase_callback(trainer, aug_increase_rate=DYNAMIC_AUG_INC_RATE,
                                   max_limit=DYNAMIC_AUG_MAX_THRESHOLD):
    step = trainer.epoch + 1
    loader = trainer.train_loader
    if step % DYNAMIC_AUG_INC_INTERVAL == 0:
        dataset = get_dataset(loader)
        dataset.augmentation_threshold = min(dataset.augmentation_threshold * aug_increase_rate, max_limit)
        print(f"Increasing random data augmentation probability by {aug_increase_rate} to {dataset.augmentation_threshold}")

