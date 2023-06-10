import sys

sys.path.insert(0, '..')
from config import DYNAMIC_DROP_OUT_REDUCTION_RATE, DYNAMIC_DROP_OUT_MAX_THRESHOLD, \
    DYNAMIC_DROP_OUT_REDUCTION_INTERVAL, DYNAMIC_AUG_INC_RATE, DYNAMIC_AUG_MAX_THRESHOLD, DYNAMIC_AUG_INC_INTERVAL
from dl_utils import get_dataset
import torch.nn as nn


def dropout_callback(trainer, dropout_rate=DYNAMIC_DROP_OUT_REDUCTION_RATE,
                     max_dropout=DYNAMIC_DROP_OUT_MAX_THRESHOLD):
    """
    A callback function designed to increase the dropout rate of the model in training after a certain number of
    epochs. The dropout rate is a regularization technique which helps in preventing overfitting during the training
    process.

    The rate of dropout is increased every few epochs based on the config parameter (in config.py)
    'DYNAMIC_DROP_OUT_REDUCTION_INTERVAL' until a maximum threshold defined by 'max_dropout'. This function is
    usually called after each epoch in the training process.

    Args:
        trainer: The object that contains the model and handles the training process. dropout_rate: The rate at
        which the dropout rate is increased. Default is value of 'DYNAMIC_DROP_OUT_REDUCTION_RATE' from config.
        max_dropout: The maximum limit to which dropout can be increased. Default is value of
        'DYNAMIC_DROP_OUT_MAX_THRESHOLD' from config.

    Functionality:
        Increases the dropout rate of all nn.Dropout modules in the model after certain number of epochs defined by
        'DYNAMIC_DROP_OUT_REDUCTION_INTERVAL'.

    :rtype: None
    :param trainer: Trainer object handling the training process
    :param dropout_rate: Rate at which to increase the dropout rate
    :param max_dropout: Maximum allowable dropout rate
    """
    step = trainer.epoch + 1
    model = trainer.model
    if step % DYNAMIC_DROP_OUT_REDUCTION_INTERVAL == 0:
        print(f"Increasing dropout rate by {dropout_rate}")
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(module.p * dropout_rate, max_dropout)


def augmentation_increase_callback(trainer, aug_increase_rate=DYNAMIC_AUG_INC_RATE,
                                   max_limit=DYNAMIC_AUG_MAX_THRESHOLD):
    """
    A callback function designed to increase the probability of data augmentation applied on the dataset during the
    training process. Data augmentation is a technique that can generate new training samples by applying
    transformations to the existing data.

    The increase in data augmentation is performed every few epochs based on 'DYNAMIC_AUG_INC_INTERVAL' until it
    reaches a specified maximum limit.

    Args:
        trainer: The object that contains the model and handles the training process.
        aug_increase_rate: The rate at which data augmentation probability is increased. Default is value of
        'DYNAMIC_AUG_INC_RATE' from config.
        max_limit: The maximum limit to which data augmentation probability can be increased.
        Default is value of 'DYNAMIC_AUG_MAX_THRESHOLD' from config.

        Functionality:
            Increases the probability of data augmentation applied on the dataset after certain number of epochs defined by 'DYNAMIC_AUG_INC_INTERVAL'.

        :rtype: None
        :param trainer: Trainer object handling the training process
        :param aug_increase_rate: Rate at which to increase the data augmentation probability
        :param max_limit: Maximum allowable data augmentation probability
    """
    step = trainer.epoch + 1
    loader = trainer.train_loader
    if step % DYNAMIC_AUG_INC_INTERVAL == 0:
        dataset = get_dataset(loader)
        dataset.augmentation_threshold = min(dataset.augmentation_threshold * aug_increase_rate, max_limit)
        print(
            f"Increasing random data augmentation probability by {aug_increase_rate} to {dataset.augmentation_threshold}")

