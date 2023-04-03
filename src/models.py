import torch.nn as nn
import torch

import pytorch_lightning as pl
from torchmetrics.classification import accuracy
from config import *


class LSTM_BASELINE_Model(nn.Module):
    def __init__(self, n_features=188, n_classes=250, n_hidden=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]

        return self.fc(out)


class LSTM_Predictor(pl.LightningModule):
    def __init__(self,
                 n_features: int = 188,
                 n_classes: int = 250,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        self.model = LSTM_BASELINE_Model(
            n_features=n_features,
            n_classes=n_classes,
            num_layers=num_layers,
            dropout=dropout
        )
        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = accuracy.Accuracy(
            task="multiclass",
            num_classes=n_classes
        )

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        landmarks = batch["landmarks"]
        labels = batch["target"]

        # forward pass through the model
        out = self(landmarks)

        #calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels

        y_hat = torch.argmax(out, dim=1)
        step_accuracy = self.accuracy(y_hat, labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "train_accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)



class TransformerPredictor(LSTM_Predictor):
    def __init__(self,seq_length:int =150, hidden_size:int=188,num_classes:int = 250):
        super().__init__()
        self.model = TransformerSequenceClassifier(input_dim =seq_length,  hidden_size = hidden_size, num_classes = num_classes)
        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = accuracy.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )

    def forward(self,x):

        x_padded = torch.nn.functional.pad(x, [0, 68], mode='constant', value=0, )
        y_hat = self.model(x_padded)
        return y_hat
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)







# Added Transformer Model
class TransformerSequenceClassifier(nn.Module):
    def __init__(self, input_dim , num_classes, num_layers=2, hidden_size=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout),
            num_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        # Permute the input sequence to match the expected format of the Transformer
        inputs = inputs.permute(1, 0, 2)

        # Pass the input sequence through the Transformer layers
        transformed = self.transformer(inputs)

        # Take the mean of the transformed sequence over the time dimension
        pooled = torch.mean(transformed, dim=0)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output

