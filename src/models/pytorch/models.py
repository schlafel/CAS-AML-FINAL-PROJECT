import sys
sys.path.insert(0,"./..")

from config import DEVICE, N_CLASSES

import torch
import torch.nn as nn
from torchmetrics.classification import accuracy


class BaseModel(nn.Module):
    def __init__(self,learning_rate,n_classes=N_CLASSES):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = accuracy.Accuracy(
            task="multiclass",
            num_classes=n_classes
        )
        self.metrics = {"train": [], "val": [], "test": []}

        self.learning_rate = learning_rate
        self.optimizer = None
        self.scheduler = None

    def calculate_accuracy(self, y_hat, y):
        preds = torch.argmax(y_hat, dim=1)
        targets = y.view(-1)
        acc = self.accuracy(preds, targets)
        return acc.cpu()

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch):
        landmarks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        # forward pass through the model
        out = self(landmarks)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels
        loss.backward()

        step_accuracy = self.calculate_accuracy(out, labels)

        del landmarks, labels

        return loss.cpu().detach(), step_accuracy.cpu()

    def validation_step(self, batch):

        with torch.no_grad():
            landmarks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # forward pass through the model
            out = self(landmarks)

            # calculate loss
            loss = 0
            if labels is not None:
                loss = self.criterion(out, labels.view(-1))

            step_accuracy = self.calculate_accuracy(out, labels)

            del landmarks, labels

        return loss.cpu().detach(), step_accuracy.cpu()

    def test_step(self, batch):
        with torch.no_grad():
            landmarks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # forward pass through the model
            out = self(landmarks)

            # calculate loss
            loss = 0
            if labels is not None:
                loss = self.criterion(out, labels.view(-1))  # need to "flatten" the labels

            preds = torch.argmax(out, dim=1)

            step_accuracy = self.calculate_accuracy(out, labels)

            del landmarks, labels

        return loss.cpu().detach(), step_accuracy.cpu(), preds.cpu()

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_mode(self):
        self.optimizer.zero_grad()
        self.train()

    def eval_mode(self):
        self.eval()

    def step_scheduler(self):
        self.scheduler.step()


class TransformerSequenceClassifier(nn.Module):
    """Transformer-based Sequence Classifier"""
    DEFAULTS = dict(
        d_model=256,
        n_head=8,
        dim_feedforward=512,
        dropout=0.1,
        layer_norm_eps=1e-5,
        norm_first=True,
        batch_first=False,
        num_layers=2,
        num_classes=N_CLASSES,
        learning_rate=0.001
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}

        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.settings['d_model'],
                nhead=self.settings['n_head'],
                dim_feedforward=self.settings['dim_feedforward'],
                dropout=self.settings['dropout'],
                layer_norm_eps=self.settings['layer_norm_eps'],
                norm_first=self.settings['norm_first'],
                batch_first=self.settings['batch_first'],
                activation='gelu'
            ),
            num_layers=self.settings['num_layers'],
            norm=nn.LayerNorm(self.settings['d_model'])
        ).to(DEVICE)

        # Output layer
        self.output_layer = nn.Linear(self.settings['d_model'], self.settings['num_classes'])

    @property
    def batch_first(self):
        return self.settings['batch_first']

    def forward(self, inputs):
        """Forward pass through the model"""
        # Check input shape
        if len(inputs.shape) != 4:
            raise ValueError(f'Expected input of shape (batch_size, seq_length, height, width), got {inputs.shape}')

        # Flatten the last two dimensions
        batch_size, seq_length, height, width = inputs.shape
        inputs = inputs.view(batch_size, seq_length, height * width).to(DEVICE)

        # Pass the input sequence through the Transformer layers
        transformed = self.transformer(inputs.to(torch.float32))

        # Take the mean of the transformed sequence over the time dimension
        pooled = torch.mean(transformed, dim=1 if self.batch_first else 0)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output


class TransformerPredictor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"],n_classes=kwargs["num_classes"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the Transformer model
        self.model = TransformerSequenceClassifier(**kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.to(DEVICE)
        ##self.save_hyperparameters() ## TODO

    def forward(self, x):
        return self.model(x)

class LSTMClassifier(nn.Module):
    """LSTM-based Sequence Classifier"""
    DEFAULTS = dict(
        input_dim=192,
        hidden_dim=100,
        layer_dim=5,
        output_dim=N_CLASSES,
        learning_rate=0.001
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}

        # LSTM
        self.lstm = nn.LSTM(self.settings['input_dim'], self.settings['hidden_dim'],
                            self.settings['layer_dim'], batch_first=True)

        # Readout layer
        self.output_layer = nn.Linear(self.settings['hidden_dim'], self.settings['output_dim'])

    def forward(self, x):
        """Forward pass through the model"""
        # Check input shape
        if len(x.shape) != 3:
            raise ValueError(f'Expected input of shape (batch_size, seq_length, input_dim), got {x.shape}')

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.settings['layer_dim'], x.size(0), self.settings['hidden_dim']).to(DEVICE)

        # Initialize cell state
        c0 = torch.zeros(self.settings['layer_dim'], x.size(0), self.settings['hidden_dim']).to(DEVICE)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.output_layer(out[:, -1, :])
        return out


class LSTMPredictor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"], n_classes=kwargs["output_dim"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the LSTM model
        self.model = LSTMClassifier(**kwargs).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.to(DEVICE)

    def forward(self, x):
        return self.model(x.to(DEVICE))
