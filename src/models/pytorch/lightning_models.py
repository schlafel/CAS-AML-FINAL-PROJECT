import sys
sys.path.insert(0,"./..")

from config import N_CLASSES, DEVICE

from torchmetrics.classification import accuracy
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LightningBaseModel(pl.LightningModule):
    def __init__(self, learning_rate, n_classes=N_CLASSES):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = accuracy.Accuracy(
            task = "multiclass",
            num_classes = n_classes
        )

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        landmarks,labels = batch

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

        landmarks,labels = batch

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
        landmarks,labels = batch

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


class LightningTransformerSequenceClassifier(nn.Module):
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
        )

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
        inputs = inputs.view(batch_size, seq_length, height * width)

        # Pass the input sequence through the Transformer layers
        transformed = self.transformer(inputs.to(torch.float32))

        # Take the mean of the transformed sequence over the time dimension
        pooled = torch.mean(transformed, dim=1 if self.batch_first else 0)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output


class LightningTransformerPredictor(LightningBaseModel):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"],n_classes=kwargs["num_classes"])

        # Instantiate the Transformer model
        self.model = LightningTransformerSequenceClassifier(**kwargs)

        #self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)