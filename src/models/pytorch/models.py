"""
This module defines a PyTorch `BaseModel` providing a basic framework for learning and validating from `Trainer`
module, from which other pytorch models are inherited. This module includes several model classes that build upon the
PyTorch's nn.Module for constructing pytorch LSTM or Transformer based models:

.. list-table:: Model Classes
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - `TransformerSequenceClassifier`
     - This is a transformer-based sequence classification model. The class constructs a transformer encoder based on user-defined parameters or default settings. The forward method first checks and reshapes the input, then passes it through the transformer layers. It then pools the sequence by taking the mean over the time dimension, and finally applies the output layer to generate the class predictions.
   * - `TransformerPredictor`
     - A TransformerPredictor model that extends the Pytorch BaseModel. This class wraps `TransformerSequenceClassifier` model and provides functionality to use it for making predictions.
   * - `MultiHeadSelfAttention`
     - This class applies a multi-head attention mechanism. It has options for causal masking and layer normalization. The input is expected to have dimensions [batch_size, seq_len, features].
   * - `TransformerBlock`
     - This class represents a single block of a transformer architecture, including multi-head self-attention and a feed-forward neural network, both with optional layer normalization and dropout. The input is expected to have dimensions [batch_size, seq_len, features].
   * - `YetAnotherTransformerClassifier`
     - This class constructs a transformer-based classifier with a specified number of `TransformerBlock` instances. The output of the model is a tensor of logits with dimensions [batch_size, num_classes].
   * - `YetAnotherTransformer`
     - This class is a wrapper for `YetAnotherTransformerClassifier` which includes learning rate, optimizer, and learning rate scheduler settings. It extends from the `BaseModel` class.
   * - `YetAnotherEnsemble`
     - This class constructs an ensemble of `YetAnotherTransformerClassifier` instances, where the outputs are concatenated and passed through a fully connected layer. This class also extends from the `BaseModel` class and includes learning rate, optimizer, and learning rate scheduler settings.

"""

import sys

sys.path.insert(0, "./..")

from config import DEVICE, N_CLASSES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import accuracy, F1Score, Precision, Recall, AUROC
from torchvision import models


"""
BaseModel
---------
"""
class BaseModel(nn.Module):
    """

    A BaseModel that extends the nn.Module from PyTorch.

    Functionality:
    #. The class initializes with a given learning rate and number of classes.
    #. It sets up the loss criterion, accuracy metric, and default states for optimizer and scheduler.
    #. It defines an abstract method 'forward' which should be implemented in the subclass.
    #. It also defines various utility functions like calculating accuracy, training, validation and testing steps, scheduler stepping, and model checkpointing.

    Args:
        learning_rate (float): The initial learning rate for optimizer.
        n_classes (int): The number of classes for classification.

    :param learning_rate: The initial learning rate for optimizer.
    :type learning_rate: float
    :param n_classes: The number of classes for classification.
    :type n_classes: int

    :returns: None
    :rtype: None

    .. note::
        The class does not directly initialize the optimizer and scheduler. They should be initialized in the subclass if needed.

    .. warning::
        The 'forward' function must be implemented in the subclass, else it will raise a NotImplementedError.

    """
    def __init__(self, learning_rate, n_classes=N_CLASSES):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = accuracy.Accuracy(
            task="multiclass",
            num_classes=n_classes
        )
        self.precision = Precision(
            task="multiclass",
            num_classes=n_classes,
            average="macro"
        )

        self.recall = Recall(
            task="multiclass",
            num_classes=n_classes,
            average="macro"

        )
        self.f1score = F1Score(
            task="multiclass",
            num_classes=n_classes,
            average="macro"

        )
        self.auroc = AUROC(
            task="multiclass",
            num_classes=n_classes,
            average="macro"

        )

        self.metrics = {"train": [], "val": [], "test": []}

        self.learning_rate = learning_rate
        self.optimizer = None
        self.scheduler = None

    def calculate_accuracy(self, y_hat, y):
        """
        Calculates the accuracy of the model's prediction.

        :param y_hat: The predicted output from the model.
        :type y_hat: Tensor
        :param y: The ground truth or actual labels.
        :type y: Tensor

        :returns: The calculated accuracy.
        :rtype: Tensor

        """
        # Damn Mac https://github.com/pytorch/pytorch/issues/92311
        preds = torch.argmax(y_hat.cpu(), dim=1)
        targets = y.view(-1).cpu()
        acc = self.accuracy(preds, targets)
        return acc

    def calculate_recall(self,y_hat,y):
        """
        Calculates the recall of the model's prediction.

        :param y_hat: The predicted output from the model.
        :type y_hat: Tensor
        :param y: The ground truth or actual labels.
        :type y: Tensor

        :returns: The calculated recall.
        :rtype: Tensor

        """
        # Damn Mac https://github.com/pytorch/pytorch/issues/92311
        preds = torch.argmax(y_hat.cpu(), dim=1)
        targets = y.view(-1).cpu()
        rec = self.recall.cpu()(preds, targets)
        return rec.cpu().numpy()

    def calculate_auc(self, y_hat, y):
        """
        Calculates the auc of the model's prediction.

        :param y_hat: The predicted output from the model.
        :type y_hat: Tensor
        :param y: The ground truth or actual labels.
        :type y: Tensor

        :returns: The calculated recall.
        :rtype: Tensor

        """
        # Damn Mac https://github.com/pytorch/pytorch/issues/92311
        preds = torch.argmax(y_hat.cpu(), dim=1)
        targets = y.view(-1).cpu()
        auc = self.auroc.cpu()(preds, targets)
        return auc.cpu().numpy()

    def calculate_precision(self,y_hat,y):
        """
        Calculates the precision of the model's prediction.

        :param y_hat: The predicted output from the model.
        :type y_hat: Tensor
        :param y: The ground truth or actual labels.
        :type y: Tensor

        :returns: The calculated precision.
        :rtype: Tensor

        """
        # Damn Mac https://github.com/pytorch/pytorch/issues/92311
        preds = torch.argmax(y_hat.cpu(), dim=1)
        targets = y.view(-1).cpu()
        prec = self.precision.cpu()(preds, targets)
        return prec.cpu().numpy()

    def calculate_f1score(self, y_hat, y):
        """
        Calculates the F1-Score of the model's prediction.

        :param y_hat: The predicted output from the model.
        :type y_hat: Tensor
        :param y: The ground truth or actual labels.
        :type y: Tensor

        :returns: The calculated f1.
        :rtype: Tensor

        """
        # Damn Mac https://github.com/pytorch/pytorch/issues/92311
        preds = torch.argmax(y_hat.cpu(), dim=1)
        targets = y.cpu().view(-1)
        f1 =self.f1score.cpu()(preds, targets)
        return f1.cpu().numpy()

    def forward(self, x):
        """
        The forward function for the BaseModel.

        :param x: The inputs to the model.
        :type x: Tensor

        :returns: None

        .. warning::
            This function must be implemented in the subclass, else it raises a NotImplementedError.

        """
        raise NotImplementedError()

    def training_step(self, batch):
        """
        Performs a training step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss and accuracy, labels and predictions
        :rtype: tuple

        """
        landmarks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

        # forward pass through the model
        predictions = self(landmarks)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.criterion(predictions, labels.view(-1))  # need to "flatten" the labels
        loss.backward()

        step_accuracy = self.calculate_accuracy(predictions, labels)

        # del landmarks, labels # @Asad: why did you delete the landmarks/labels

        return loss.cpu().detach(), step_accuracy.cpu(), labels.cpu(), predictions.cpu().detach()

    def validation_step(self, batch):
        """
        Performs a validation step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss and accuracy, labels and predictions
        :rtype: tuple

        """

        with torch.no_grad():
            landmarks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # forward pass through the model
            predictions = self(landmarks)

            # calculate loss
            loss = 0
            if labels is not None:
                loss = self.criterion(predictions, labels.view(-1))

            step_accuracy = self.calculate_accuracy(predictions, labels)

            # del landmarks, labels #

        return loss.cpu().detach(), step_accuracy.cpu(), labels.cpu(), predictions.cpu()

    def test_step(self, batch):
        """
        Performs a test step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss, accuracy, labels, and model predictions.
        :rtype: tuple

        """
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

            # del landmarks, labels

        return loss.cpu().detach(), step_accuracy.cpu(), labels.cpu(), out.cpu()

    def optimize(self):
        """
        Steps the optimizer and sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_mode(self):
        """
        Sets the model to training mode.
        """
        self.optimizer.zero_grad()
        self.train()

    def eval_mode(self):
        """
        Sets the model to evaluation mode.
        """
        self.eval()

    def step_scheduler(self):
        """
        Steps the learning rate scheduler, adjusting the optimizer's learning rate as necessary.
        """
        self.scheduler.step()

    def get_lr(self):
        """
        Gets the current learning rate of the model.

        :returns: The current learning rate.
        :rtype: float
        """
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, filepath):
        """
        Saves the model and optimizer states to a checkpoint.

        :param filepath: The file path where to save the model checkpoint.
        :type filepath: str
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Loads the model and optimizer states from a checkpoint.

        :param filepath: The file path where to load the model checkpoint from.
        :type filepath: str
        """
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class TransformerSequenceClassifier(nn.Module):
    """
    =============================
    TransformerSequenceClassifier
    =============================

    A Transformer-based Sequence Classifier. This class utilizes a transformer encoder to process the input sequence.

    The transformer encoder consists of a stack of N transformer layers that are applied to the input sequence.
    The output sequence from the transformer encoder is then passed through a linear layer to generate class predictions.

    Attributes
    ----------
    DEFAULTS : dict
        Default settings for the transformer encoder and classifier. These can be overridden by passing values in the constructor.
    transformer : nn.TransformerEncoder
        The transformer encoder used to process the input sequence.
    output_layer : nn.Linear
        The output layer used to generate class predictions.
    batch_first : bool
        Whether the first dimension of the input tensor represents the batch size.

    Methods
    -------
    forward(inputs)
        Performs a forward pass through the model.
    """
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
    """
    ===================
    TransformerPredictor
    ===================

    A TransformerPredictor model that extends the Pytorch BaseModel.

    This class wraps the TransformerSequenceClassifier model and provides functionality to use it for making predictions.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    model : TransformerSequenceClassifier
        The transformer sequence classifier used for making predictions.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"], n_classes=kwargs["num_classes"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the Transformer model
        self.model = TransformerSequenceClassifier(**kwargs)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=kwargs["gamma"])

        self.to(DEVICE)
        ##self.save_hyperparameters() ## TODO

    def forward(self, x):
        return self.model(x)


class LSTMClassifier(nn.Module):
    """
    ===========================
    LSTMClassifier
    ===========================

    A LSTM-based Sequence Classifier. This class utilizes a LSTM network for sequence classification tasks.

    Attributes
    ----------
    DEFAULTS : dict
        Default settings for the LSTM and classifier. These can be overridden by passing values in the constructor.
    lstm : nn.LSTM
        The LSTM network used for processing the input sequence.
    dropout : nn.Dropout
        The dropout layer applied after LSTM network.
    output_layer : nn.Linear
        The output layer used to generate class predictions.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
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
    """
    ===================
    LSTMPredictor
    ===================

    A LSTMPredictor model that extends the Pytorch BaseModel.

    This class wraps the LSTMClassifier model and provides functionality to use it for making predictions.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    model : LSTMClassifier
        The LSTM classifier used for making predictions.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"], n_classes=kwargs["output_dim"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the LSTM model
        self.model = LSTMClassifier(**kwargs).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=kwargs["gamma"])

        self.to(DEVICE)

    def forward(self, x):
        return self.model(x.to(DEVICE))


class HybridModel(BaseModel):
    """
    ===================
    HybridModel
    ===================

    A HybridModel that extends the Pytorch BaseModel.

    This class combines the LSTMClassifier and TransformerSequenceClassifier models
    and provides functionality to use the combined model for making predictions.

    Attributes
    ----------
    lstm : LSTMClassifier
        The LSTM classifier used for making predictions.
    transformer : TransformerSequenceClassifier
        The transformer sequence classifier used for making predictions.
    fc : nn.Linear
        The final fully-connected layer.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
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
    """
    ===================
    TransformerEnsemble
    ===================

    A TransformerEnsemble that extends the Pytorch BaseModel.

    This class creates an ensemble of TransformerSequenceClassifier models
    and provides functionality to use the ensemble for making predictions.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    models : nn.ModuleList
        The list of transformer sequence classifiers used for making predictions.
    fc : nn.Linear
        The final fully-connected layer.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
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
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=common_params["gamma"])

        self.to(DEVICE)
        ##self.save_hyperparameters() ## TODO

    def forward(self, x):
        model_outputs = [model(x) for model in self.models]
        combined = torch.cat(model_outputs, dim=1)
        output = self.fc(combined)
        return output


class HybridEnsembleModel(BaseModel):
    """
    =======================
    HybridEnsembleModel
    =======================

    A HybridEnsembleModel that extends the Pytorch BaseModel.

    This class creates an ensemble of LSTM and Transformer models and provides
    functionality to use the ensemble for making predictions.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    lstms : nn.ModuleList
        The list of LSTM models.
    models : nn.ModuleList
        The list of Transformer models.
    fc : nn.Linear
        The final fully-connected layer.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
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

        self.models = nn.ModuleList([TransformerSequenceClassifier(num_layers=i + 1,
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


class CVTransferLearningModel(BaseModel):
    """
    ========================
    CVTransferLearningModel
    ========================

    A CVTransferLearningModel that extends the Pytorch BaseModel.

    This class applies transfer learning for computer vision tasks using pretrained models.
    It also provides a forward method to pass an input through the model.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    model : nn.Module
        The base model for transfer learning.
    optimizer : torch.optim.Adam
        The optimizer used for updating the model parameters.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler used for adapting the learning rate during training.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.
    """
    DEFAULTS = dict({})

    def __init__(self, **kwargs):

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}
        super().__init__(learning_rate=self.settings['hparams']['learning_rate'])

        # get weights
        if "weights" not in self.settings['hparams'].keys():
            self.settings['hparams']['weights'] = None

        model = models.get_model(self.settings['hparams']['backbone'],
                                 weights=self.settings['hparams']['weights'],
                                 )

        # Freeze layers
        # Freeze the parameters....
        if self.settings['hparams']['weights'] is not None :
            for p in model.parameters():
                if hasattr(p, "requires_grad"):
                    p.requires_grad = False

        # recursively iterate over child modules until we find the last fully connected layer
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                last_layer = module
                last_layer_name = name
            if isinstance(module, nn.Sequential):
                for name2, module2 in module.named_children():
                    if isinstance(module2, nn.Linear):
                        last_layer = module2
                        last_layer_name = name + "." + name2

        new_last_layer = nn.Linear(last_layer.in_features,
                                   self.settings['params']['num_classes'],
                                   bias=True,
                                   )

        setattr(model, last_layer_name, new_last_layer)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=kwargs['hparams']["gamma"])

        self.model = model

        self.to(DEVICE)

    def forward(self, x):
        # reshape inputs?
        x_new = F.pad(x, (0, 1), value=0.).reshape(-1, 64, 48, 3).moveaxis(-1, 1)
        # pass it to the model
        return self.model(x_new.to(DEVICE))


class MultiHeadSelfAttention(nn.Module):
    """
    ======================
    MultiHeadSelfAttention
    ======================

    A MultiHeadSelfAttention module that extends the nn.Module from PyTorch.

    Functionality:
    #. The class initializes with a given dimension size, number of attention heads, dropout rate, layer normalization and causality.
    #. It sets up the multihead attention module and layer normalization.
    #. It also defines a forward method that applies the multihead attention, causal masking if requested, and layer normalization if requested.

    Attributes
    ----------
    multihead_attn : nn.MultiheadAttention
        The multihead attention module.
    layer_norm : nn.LayerNorm or None
        The layer normalization module. If it is not applied, set to None.
    causal : bool
        If True, applies causal masking.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.

    Args:
        dim (int): The dimension size of the input data.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        layer_norm (bool): Whether to apply layer normalization.
        causal (bool): Whether to apply causal masking.

    Returns: None
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, layer_norm=True, causal=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else None
        self.causal = causal

    def forward(self, x):
        x = x.permute(1, 0, 2)

        # Apply a causal mask if requested
        if self.causal:
            seq_len = x.size(0)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        else:
            mask = None

        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)

        # Apply Layer Normalization if requested
        if self.layer_norm is not None:
            attn_output = self.layer_norm(attn_output)

        return attn_output.permute(1, 0, 2)


class TransformerBlock(nn.Module):
    """
    ================
    TransformerBlock
    ================

    A TransformerBlock module that extends the nn.Module from PyTorch.

    Functionality:
    #. The class initializes with a given dimension size, number of attention heads, expansion factor, attention dropout rate, and dropout rate.
    #. It sets up the multihead self-attention module, layer normalization and feed-forward network.
    #. It also defines a forward method that applies the multihead self-attention, dropout, layer normalization and feed-forward network.

    Attributes
    ----------
    norm1, norm2, norm3 : nn.LayerNorm
        The layer normalization modules.
    attn : MultiHeadSelfAttention
        The multihead self-attention module.
    feed_forward : nn.Sequential
        The feed-forward network.
    dropout : nn.Dropout
        The dropout module.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.

    Args:
        dim (int): The dimension size of the input data.
        num_heads (int): The number of attention heads.
        expansion_factor (int): The expansion factor for the hidden layer size in the feed-forward network.
        attn_dropout (float): The dropout rate for the attention module.
        drop_rate (float): The dropout rate for the module.

    Returns: None
    """
    def __init__(self, dim=192, num_heads=4, expansion_factor=4, attn_dropout=0.2, drop_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=attn_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim),
            nn.GELU(),
            nn.Linear(expansion_factor * dim, dim),
            nn.Dropout(drop_rate),
            nn.Linear(dim, expansion_factor * dim),
            nn.GELU(),
            nn.Linear(expansion_factor * dim, dim),
            nn.Dropout(drop_rate),
        )

        self.dropout = nn.Dropout(drop_rate)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        # Apply norm and self attention
        x = self.attn(self.norm1(x)) + x
        x = self.dropout(x)

        # Apply norm and feed forward network
        res = x
        x = self.norm2(x)
        x = self.feed_forward(x) + res
        x = self.dropout(x)
        x = self.norm3(x)

        return x


class YetAnotherTransformerClassifier(nn.Module):
    """
    ===============================
    YetAnotherTransformerClassifier
    ===============================

    A YetAnotherTransformerClassifier module that extends the nn.Module from PyTorch.

    Functionality:
    #. The class initializes with a set of parameters for the transformer blocks.
    #. It sets up the transformer blocks and the output layer.
    #. It also defines a forward method that applies the transformer blocks, takes the mean over the time dimension of the transformed sequence, and applies the output layer.

    Attributes
    ----------
    DEFAULTS : dict
        The default settings for the transformer.
    settings : dict
        The settings for the transformer, with any user-provided values overriding the defaults.
    transformer : nn.ModuleList
        The list of transformer blocks.
    output_layer : nn.Linear
        The output layer.

    Methods
    -------
    forward(inputs)
        Performs a forward pass through the model.

    Args:
        kwargs (dict): A dictionary containing the parameters for the transformer blocks.

    Returns: None
    """
    DEFAULTS = dict(
        d_model=192,
        n_head=8,
        expand=4,
        drop_rate=0.0001,
        attn_dropout=0.1,
        num_layers=2,
        num_classes=N_CLASSES,
        learning_rate=0.001
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}

        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=self.settings['d_model'],
                num_heads=self.settings['n_head'],
                expansion_factor=self.settings['expand'],
                drop_rate=self.settings['drop_rate'],
                attn_dropout=self.settings['attn_dropout']
            ) for _ in range(self.settings['num_layers'])
        ])

        # Output layer
        self.output_layer = nn.Linear(self.settings['d_model'], self.settings['num_classes'])

    def forward(self, inputs):
        """Forward pass through the model"""
        # Check input shape
        if len(inputs.shape) != 4:
            raise ValueError(f'Expected input of shape (batch_size, seq_length, height, width), got {inputs.shape}')

        # Flatten the last two dimensions
        batch_size, seq_length, height, width = inputs.shape
        inputs = inputs.view(batch_size, seq_length, height * width).to(DEVICE)

        # Pass the input sequence through the Transformer layers
        for transformer_block in self.transformer:
            inputs = transformer_block(inputs)

        # Take the mean of the transformed sequence over the time dimension
        pooled = torch.mean(inputs, dim=1)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output


class YetAnotherTransformer(BaseModel):
    """
    =====================
    YetAnotherTransformer
    =====================

    A YetAnotherTransformer model that extends the Pytorch BaseModel.

    Functionality:
    #. The class initializes with a set of parameters for the YetAnotherTransformerClassifier.
    #. It sets up the YetAnotherTransformerClassifier model, the optimizer and the learning rate scheduler.
    #. It also defines a forward method that applies the YetAnotherTransformerClassifier model.

    Attributes
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    model : YetAnotherTransformerClassifier
        The YetAnotherTransformerClassifier model.
    optimizer : torch.optim.AdamW
        The AdamW optimizer.
    scheduler : torch.optim.lr_scheduler.ExponentialLR
        The learning rate scheduler.

    Methods
    -------
    forward(x)
        Performs a forward pass through the model.

    Args:
        kwargs (dict): A dictionary containing the parameters for the YetAnotherTransformerClassifier, optimizer and learning rate scheduler.

    Returns: None
    """
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_params = kwargs['YetAnotherTransformerClassifier']

        super().__init__(learning_rate=common_params["learning_rate"], n_classes=common_params["num_classes"])

        self.learning_rate = common_params["learning_rate"]

        # Instantiate the Transformer model
        self.model = YetAnotherTransformerClassifier(**transformer_params)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=common_params["gamma"])

        self.to(DEVICE)

    def forward(self, x):
        return self.model(x)


class YetAnotherEnsemble(BaseModel):
    """
    ==================
    YetAnotherEnsemble
    ==================

    A YetAnotherEnsemble model that extends the Pytorch BaseModel.

    Functionality:
    #. The class initializes with a set of parameters for the YetAnotherTransformerClassifier.
    #. It sets up an ensemble of YetAnotherTransformerClassifier models, a fully connected layer, the optimizer and the learning rate scheduler.
    #. It also defines a forward method that applies each YetAnotherTransformerClassifier model in the ensemble, concatenates the outputs and applies the fully connected layer.

    Args:
        kwargs (dict): A dictionary containing the parameters for the YetAnotherTransformerClassifier models, fully connected layer, optimizer and learning rate scheduler.

    Returns: None
    """
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_params = kwargs['YetAnotherTransformerClassifier']

        n_models = common_params["n_models"]
        super().__init__(learning_rate=common_params["learning_rate"], n_classes=common_params["num_classes"])

        self.learning_rate = common_params["learning_rate"]

        # Ensemble
        self.models = nn.ModuleList([YetAnotherTransformerClassifier(num_layers=2 + i,
                                                                     **transformer_params) for i, _ in
                                     enumerate(range(n_models))])

        self.fc = nn.Linear(common_params["num_classes"] * n_models, common_params["num_classes"]).to(DEVICE)

        self.optimizer = torch.optim.AdamW(self.models.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=common_params["gamma"])

        self.to(DEVICE)

    def forward(self, x):
        model_outputs = [model(x) for model in self.models]
        combined = torch.cat(model_outputs, dim=1)
        output = self.fc(combined)
        return output


