import sys

sys.path.insert(0, '..')
from config import DYNAMIC_DROP_OUT_INIT_RATE, DYNAMIC_DROP_OUT_REDUCTION_RATE, DYNAMIC_DROP_OUT_MAX_THRESHOLD, DYNAMIC_DROP_OUT_REDUCTION_INTERVAL
import torch.nn as nn


def dropout_callback(step, model, dropout_rate=DYNAMIC_DROP_OUT_REDUCTION_RATE,
                     max_dropout=DYNAMIC_DROP_OUT_MAX_THRESHOLD):
    if step % DYNAMIC_DROP_OUT_REDUCTION_INTERVAL == 0:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(module.p * dropout_rate, max_dropout)
