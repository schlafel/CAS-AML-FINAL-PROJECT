import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pytorch_lightning as pl
from torchmetrics.classification import accuracy
import sys
sys.path.insert(0,"./..")
from config import *

import math


class LSTM_BASELINE_Model(nn.Module):
    def __init__(self, n_features=N_LANDMARKS, n_classes=N_CLASSES, n_hidden=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.hidden_size = n_hidden
        self.num_layers = num_layers

        input_size = n_features * 2  # 2 is for x and y coordinates

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.fc = nn.Linear(n_hidden, n_classes)

    def __repr__(self):
        return "LSTM_BASELINE_Model"

    def forward(self, x, seq_lengths):
        batch_size, seq_len, landmarks, coords = x.size()
        x = x.view(batch_size, seq_len, -1).float()

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Set the initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()
        c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()

        out, _ = self.lstm(x, (h0, c0))

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.fc(out[:, -1, :])

        return out


class ImprovedLSTMModel(nn.Module):
    def __init__(self, n_features=N_LANDMARKS, n_classes=N_CLASSES, n_hidden=512, num_layers=3, dropout=0.5):
        super().__init__()

        self.hidden_size = n_hidden
        self.num_layers = num_layers

        input_size = n_features * 2  # 2 is for x and y coordinates

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def __repr__(self):
        return "ImprovedLSTMModel"

    def forward(self, x, seq_lengths):
        batch_size, seq_len, landmarks, coords = x.size()
        x = x.view(batch_size, seq_len, -1).float()

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Set the initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()
        c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()

        out, _ = self.lstm(x, (h0, c0))

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Apply dropout and fully connected layers
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=150):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ASLTransformerModel(nn.Module):
    def __init__(self, n_features=N_LANDMARKS * 2, n_classes=N_CLASSES, d_model=512, nhead=8, num_layers=4,
                 dropout=0.3):
        super(ASLTransformerModel, self).__init__()

        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(d_model, n_classes)

    def __repr__(self):
        return "ASLTransformerModel"

    def forward(self, x, seq_lengths):
        batch_size, seq_len, landmarks, coords = x.size()
        x = x.view(batch_size, seq_len, -1).float()

        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x


class LSTM_Predictor(pl.LightningModule):
    def __init__(self,
                 n_features: int = 188,
                 n_classes: int = N_CLASSES,
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

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        landmarks = batch["landmarks"]
        labels = batch["target"]

        # forward pass through the model
        out = self(landmarks)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels

        y_hat = torch.argmax(out, dim=1)
        step_accuracy = self.accuracy(y_hat, labels.view(-1))

        self.train_step_outputs.append(dict({"train_accuracy": step_accuracy,
                                             "train_loss": loss}))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy,
                 prog_bar=True,
                 logger=True)
        return {"loss": loss, "train_accuracy": step_accuracy}

    def on_train_epoch_end(self) -> None:
        # get average training accuracy
        avg_loss = torch.stack([x['train_loss'] for x in self.train_step_outputs]).mean()

        train_acc = torch.stack([x['train_accuracy'] for x in self.train_step_outputs]).mean()
        print(" ")
        print(f"EPOCH {self.current_epoch}: Train accuracy: {train_acc}")
        self.train_step_outputs.clear()
        print(100*"*")

    def validation_step(self, batch, batch_idx):

        landmarks, labels = batch["landmarks"], batch["target"]

        # forward pass through the model
        out = self(landmarks)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels

        y_hat = torch.argmax(out, dim=1)
        step_accuracy = self.accuracy(y_hat, labels.view(-1))

        self.validation_step_outputs.append(
            dict({
                "val_accuracy": step_accuracy,
                "val_loss": loss,
            }))

        self.log_dict(
            dict({
                "val_loss": loss,
                "val_accuracy": step_accuracy,
            }),
            prog_bar=False,
            logger=True,
            on_epoch=True)
        return y_hat

    def on_validation_end(self):

        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()

        val_acc = torch.stack([x['val_accuracy'] for x in self.validation_step_outputs]).mean()

        self.print(f"EPOCH {self.current_epoch}, Validation Accuracy: {val_acc}")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        landmarks = batch["landmarks"]
        labels = batch["target"]

        # forward pass through the model
        out = self(landmarks)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels

        y_hat = torch.argmax(out, dim=1)
        step_accuracy = self.accuracy(y_hat, labels.view(-1))

        self.test_step_outputs.append(dict({"test_accuracy": step_accuracy,
                                            "test_loss": loss,
                                            "preds": y_hat,
                                            "target": labels.view(-1)}))

        self.log_dict(dict({"test_loss": loss,
                            "test_accuracy": step_accuracy}),
                      logger=True,
                      on_epoch=True)
        return y_hat

    def on_test_end(self) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()

        val_acc = torch.stack([x['test_accuracy'] for x in self.test_step_outputs]).mean()

        self.print(f"EPOCH {self.current_epoch}, Accuracy: {val_acc}")
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)



class TransformerPredictor(LSTM_Predictor):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=512,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 norm_first=True,
                 batch_first=True,
                 num_layers=2,
                 num_classes=250):
        super().__init__()
        self.model = TransformerSequenceClassifier(d_model=d_model,
                                                   n_head=n_head,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   layer_norm_eps=layer_norm_eps,
                                                   norm_first=norm_first,
                                                   batch_first=batch_first,
                                                   num_layers=num_layers,
                                                   num_classes=num_classes)
        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = accuracy.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.save_hyperparameters()

    def forward(self, x):
        # x_padded = torch.nn.functional.pad(x, [0, 68], mode='constant', value=0, )
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # Flatten the inputs
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)


# Added Transformer Model
class TransformerSequenceClassifier(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward = 512,
                 dropout=0.1,
                 layer_norm_eps = 1e-5,
                 norm_first = True,
                 batch_first = False,
                 num_layers=2,
                 num_classes = 250,
    ):
        super().__init__()
        self.batch_first = batch_first
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead = n_head,
                                       dim_feedforward=dim_feedforward,
                                       dropout = dropout,
                                       layer_norm_eps=layer_norm_eps,
                                       norm_first=norm_first,
                                       batch_first=batch_first),
            #norm = 1e-6,
            num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        # inputs = inputs.reshape([inputs.shape[0],inputs.shape[1],inputs.shape[2]*inputs.shape[3]])
        # Permute the input sequence to match the expected format of the Transformer

        #inputs = inputs.permute(1, 0, 2)

        # Pass the input sequence through the Transformer layers
        transformed = self.transformer(inputs.to(torch.float32))

        # Take the mean of the transformed sequence over the time dimension
        pooled = torch.mean(transformed, dim=1)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output
