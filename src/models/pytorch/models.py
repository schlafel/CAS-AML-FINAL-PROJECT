import sys

sys.path.insert(0, "./..")

from config import DEVICE, N_CLASSES

import torch
import torch.nn as nn
from torchmetrics.classification import accuracy


class BaseModel(nn.Module):
    def __init__(self, learning_rate, n_classes=N_CLASSES):
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

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


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
        super().__init__(learning_rate=kwargs["learning_rate"], n_classes=kwargs["num_classes"])

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
        learning_rate=0.001,
        dropout=0.5
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}

        # LSTM
        self.lstm = nn.LSTM(self.settings['input_dim'], self.settings['hidden_dim'],
                            self.settings['layer_dim'], dropout=self.settings['dropout'], batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(self.settings['dropout'])

        # Readout layer
        self.output_layer = nn.Linear(self.settings['hidden_dim'], self.settings['output_dim'])

    def forward(self, x):
        """Forward pass through the model"""

        # Initialize hidden state with zeros
        batch_size, seq_length, height, width = x.shape
        x = x.view(batch_size, seq_length, height * width).to(DEVICE)

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


class HybridModel(BaseModel):
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_kwargs = kwargs['transformer_params']
        lstm_kwargs = kwargs['lstm_params']

        super().__init__(learning_rate=common_params["learning_rate"], n_classes=common_params["num_classes"])

        self.lstm = LSTMClassifier(**lstm_kwargs).to(DEVICE)
        self.transformer = TransformerSequenceClassifier(**transformer_kwargs).to(DEVICE)
        self.fc = nn.Linear(common_params["num_classes"] * 2, common_params["num_classes"]).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=common_params["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.to(DEVICE)

    def forward(self, x):
        lstm_output = self.lstm(x)
        transformer_output = self.transformer(x)

        # Concatenate the outputs of LSTM and Transformer along the feature dimension
        combined = torch.cat((lstm_output, transformer_output), dim=1)

        # Pass the combined output through the final fully-connected layer
        output = self.fc(combined)

        return output


class TransformerEnsemble(BaseModel):
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_params = kwargs['TransformerSequenceClassifier']

        n_models = common_params["n_models"]
        super().__init__(learning_rate=common_params["learning_rate"], n_classes=common_params["num_classes"])

        self.learning_rate = common_params["learning_rate"]

        # Ensemble
        self.models = nn.ModuleList([TransformerSequenceClassifier(num_layers=2 + i,
                                                                   **transformer_params) for i, _ in
                                     enumerate(range(n_models))])

        self.fc = nn.Linear(common_params["num_classes"] * n_models, common_params["num_classes"]).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.to(DEVICE)
        ##self.save_hyperparameters() ## TODO

    def forward(self, x):
        model_outputs = [model(x) for model in self.models]
        combined = torch.cat(model_outputs, dim=1)
        output = self.fc(combined)
        return output


class HybridEnsembleModel(BaseModel):
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_params = kwargs['TransformerSequenceClassifier']
        lstm_kwargs = kwargs['lstm_params']

        n_models = common_params["n_models"]
        super().__init__(learning_rate=common_params["learning_rate"], n_classes=common_params["num_classes"])

        self.learning_rate = common_params["learning_rate"]

        # Ensemble
        self.lstms = nn.ModuleList([LSTMClassifier(**lstm_kwargs).to(DEVICE) for i, _ in
                                     enumerate(range(n_models))])

        self.models = nn.ModuleList([TransformerSequenceClassifier(num_layers=i,
                                                                   **transformer_params) for i, _ in
                                     enumerate(range(n_models))])

        self.fc = nn.Linear(common_params["num_classes"] * (n_models * 2), common_params["num_classes"]).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=common_params["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        self.to(DEVICE)

    def forward(self, x):
        lstm_outputs = [ltsm(x) for ltsm in self.lstms]
        l_combined = torch.cat(lstm_outputs, dim=1)

        model_outputs = [model(x) for model in self.models]
        t_combined = torch.cat(model_outputs, dim=1)

        # Concatenate the outputs of LSTM and Transformer along the feature dimension
        output = torch.cat((l_combined, t_combined), dim=1)

        # Pass the combined output through the final fully-connected layer
        output = self.fc(output)

        return output
