import torch.nn as nn
import torch

import pytorch_lightning as pl
from torchmetrics.classification import accuracy
from config import *

class LSTM_BASELINE_Model(nn.Module):
    def __init__(self, n_features, n_classes=250, n_hidden=256, num_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=.3)

        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]

        return self.fc(out)


class LSTM_Predictor(pl.LightningModule):
    def __init__(self,
                 n_features: int,
                 n_classes: int = 250,
                 num_layers: int = 3):
        super().__init__()

        self.model = LSTM_BASELINE_Model(n_features=n_features,
                                         n_classes=n_classes,
                                         num_layers=num_layers)
        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = accuracy.Accuracy(task="multiclass",
                                          num_classes=n_classes
                                          )


    def forward(self, x, labels):
        y_hat = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(y_hat, labels.view(-1)) #need to "flatten" the labels
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        landmarks = batch["landmarks"]
        labels = batch["target"]

        # forward pass through the model
        loss, out = self(landmarks, labels)
        y_hat = torch.argmax(out, dim=1)
        step_accuracy = self.accuracy(y_hat, labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "train_accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

