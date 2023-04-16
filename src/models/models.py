import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics.classification import accuracy
from config import *


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

    def forward(self, x, seq_lengths):
        batch_size, seq_len, landmarks, coords = x.size()
        x = x.view(batch_size, seq_len, -1).float()

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        # Set the initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()
        c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(DEVICE).float()

        out, _ = self.lstm(x, (h0, c0))

        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.fc(out[:, -1, :])

        return out


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

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        landmarks = batch[0]
        labels = batch[1]

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
    def __init__(self,seq_length:int =150, hidden_size:int=256,num_classes:int = 250,num_heads:int = 8):
        super().__init__()
        self.model = TransformerSequenceClassifier(input_dim =seq_length,  num_heads = num_heads,
                                                   hidden_size = hidden_size, num_classes = num_classes)
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

