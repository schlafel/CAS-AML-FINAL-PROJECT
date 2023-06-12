"""
This kodule defines a PyTorch `BaseModel` providing a basic framework for learning and validating from `Trainer`
"""
import sys

sys.path.insert(0, "./..")

from config import DEVICE, N_CLASSES

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, LayerNormalization, LSTM, Flatten
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class BaseModel(tf.keras.Model):
    """
    A BaseModel that extends the tf.keras.Model.

    Functionality:
    #. The class initializes with a given learning rate.
    #. It sets up the loss criterion, accuracy metric, and default states for optimizer and scheduler.
    #. It defines an abstract method 'call' which should be implemented in the subclass.
    #. It also defines various utility functions like calculating accuracy, training, validation and testing steps, scheduler stepping, and model checkpointing.

    Args:
        learning_rate (float): The initial learning rate for optimizer.

    :param learning_rate: The initial learning rate for optimizer.
    :type learning_rate: float

    :returns: None
    :rtype: None

    .. note::
        The class does not directly initialize the optimizer and scheduler. They should be initialized in the subclass
        if needed.

    .. warning::
        The 'call' function must be implemented in the subclass, else it will raise a NotImplementedError.
    """
    def __init__(self, learning_rate):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.recall = tf.keras.metrics.Recall()
        self.precision = tf.keras.metrics.Precision()
        self.f1score = tfa.metrics.F1Score(num_classes=N_CLASSES)
        self.f1score = tf.keras.metrics.AUC()



        self.optimizer = None
        self.scheduler = None

        self.training = False

        self.compile()

    def calculate_accuracy(self, y_pred, y_true):
        """
        Calculates the accuracy of the model's prediction.

        :param y_pred: The predicted output from the model.
        :type y_pred: Tensor
        :param y_true: The ground truth or actual labels.
        :type y_true: Tensor

        :returns: The calculated accuracy.
        :rtype: float
        """
        return self.accuracy(y_true, y_pred)
    def calculate_f1score(self, y_pred, y_true):
        """
        Calculates the F1-Score of the model's prediction.

        :param y_pred: The predicted output from the model.
        :type y_pred: Tensor
        :param y_true: The ground truth or actual labels.
        :type y_true: Tensor

        :returns: The calculated accuracy.
        :rtype: float
        """
        return self.f1score(y_true, tf.argmax(y_pred,axis = 1)).numpy()

    def calculate_auc(self, y_pred, y_true):
        """
        Calculates the AUC of the model's prediction.

        :param y_pred: The predicted output from the model.
        :type y_pred: Tensor
        :param y_true: The ground truth or actual labels.
        :type y_true: Tensor

        :returns: The calculated accuracy.
        :rtype: float
        """
        return self.auc(y_true, tf.argmax(y_pred,axis = 1)).numpy()

    def calculate_recall(self, y_pred, y_true):
        """
        Calculates the Recall of the model's prediction.

        :param y_pred: The predicted output from the model.
        :type y_pred: Tensor
        :param y_true: The ground truth or actual labels.
        :type y_true: Tensor

        :returns: The calculated accuracy.
        :rtype: float
        """
        return self.recall(y_true, tf.argmax(y_pred,axis = 1)).numpy()

    def calculate_precision(self, y_pred, y_true):
        """
        Calculates the Recall of the model's prediction.

        :param y_pred: The predicted output from the model.
        :type y_pred: Tensor
        :param y_true: The ground truth or actual labels.
        :type y_true: Tensor

        :returns: The calculated accuracy.
        :rtype: float
        """
        return self.precision(y_true, tf.argmax(y_pred,axis = 1)).numpy()


    def call(self, inputs, training=False):
        """
        The call function for the BaseModel.

        :param inputs: The inputs to the model.
        :type inputs: Tensor
        :param training: A flag indicating whether the model is in training mode. Default is False.
        :type training: bool

        :returns: None

        .. warning::
            This function must be implemented in the subclass, else it raises a NotImplementedError.
        """
        raise NotImplementedError()
    @tf.function
    def training_step(self, batch):
        """
        Performs a training step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss and accuracy, labels and predictions
        :rtype: tuple
        """
        landmarks, labels = batch

        # Forward pass

        with tf.GradientTape() as tape:
            predictions = self(landmarks, training=self.training)

            loss = 0
            if labels is not None:
                loss = self.criterion(labels, predictions)

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Calculate accuracy
        accuracy = self.calculate_accuracy(predictions, labels)

        # del landmarks, labels

        return loss, accuracy, labels, predictions
    @tf.function
    def validation_step(self, batch):
        """
        Performs a validation step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss and accuracy.
        :rtype: tuple
        """
        landmarks, labels = batch

        predictions = self(landmarks, training=False)

        loss = 0
        if labels is not None:
            loss = self.criterion(labels, predictions)

        accuracy = self.calculate_accuracy(predictions, labels)

        return loss, accuracy, labels, predictions

    @tf.function
    def test_step(self, batch):
        """
        Performs a test step using the input batch data.

        :param batch: A tuple containing input data and labels.
        :type batch: tuple

        :returns: The calculated loss, accuracy, and model predictions.
        :rtype: tuple
        """

        landmarks, labels = batch

        predictions = self(landmarks, training=self.training)

        loss = 0
        if labels is not None:
            loss = self.criterion(labels, predictions)

        preds = tf.argmax(predictions, axis=-1)
        accuracy = self.calculate_accuracy(predictions, labels)

        return loss, accuracy, labels, predictions

    def optimize(self):
        """
        Sets the model to training mode.
        """
        self.training = True

    def train_mode(self):
        """
        Sets the model to training mode.
        """
        self.training = True

    def eval_mode(self):
        """
        Sets the model to evaluation mode.
        """
        self.training = False

    def step_scheduler(self):
        """
        Adjusts the learning rate according to the learning rate scheduler.
        """
        try:
            self.optimizer.learning_rate = self.scheduler(self.optimizer.iterations)
        except:
            pass

    def get_lr(self):
        """
        Gets the current learning rate of the model.

        :returns: The current learning rate.
        :rtype: float
        """
        if not hasattr(self.optimizer, 'iterations'):
            return self.learning_rate
        return self.scheduler(self.optimizer.iterations).numpy()

    def save_checkpoint(self, filepath):
        """
        Saves the model weights to a checkpoint.

        :param filepath: The file path where to save the model checkpoint.
        :type filepath: str
        """
        self.save_weights(filepath)

    def load_checkpoint(self, filepath):
        """
        Loads the model weights from a checkpoint.

        :param filepath: The file path where to load the model checkpoint from.
        :type filepath: str
        """
        self.load_weights(filepath)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    A Transformer Encoder layer as a subclass of tf.keras.layers.Layer.

    Functionality:
    #. The class first initializes with key parameters for MultiHeadAttention and feedforward network.
    #. Then it defines the key components like multi-head attention, feedforward network, layer normalization, and dropout.
    #. In the call function, it takes input and performs self-attention, followed by layer normalization and feedforward operation.

    Args:
        d_model (int): The dimensionality of the input.
        n_head (int): The number of heads in the multi-head attention.
        dim_feedforward (int): The dimensionality of the feedforward network model.
        dropout (float): The dropout value.

    :param d_model: The dimensionality of the input.
    :type d_model: int
    :param n_head: The number of heads in the multi-head attention.
    :type n_head: int
    :param dim_feedforward: The dimensionality of the feedforward network model.
    :type dim_feedforward: int
    :param dropout: The dropout value.
    :type dropout: float

    :returns: None
    :rtype: None

    .. note::
        The implementation is based on the "Attention is All You Need" paper.

    .. warning::
        Ensure that the input dimension 'd_model' is divisible by the number of attention heads 'n_head'.
    """
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads=n_head, key_dim=int(d_model / n_head))
        self.feed_forward = tf.keras.Sequential([
            Dense(dim_feedforward, activation='gelu'),  # Use 'gelu' activation to match PyTorch version
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-5)
        self.layernorm2 = LayerNormalization(epsilon=1e-5)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    @tf.function
    def call(self, x, training=False):
        attn_output = self.self_attn(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        feedforward_output = self.feed_forward(out1)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        out2 = self.layernorm2(out1 + feedforward_output)

        return out2


class TransformerSequenceClassifier(Model):
    """
    A Transformer Sequence Classifier as a subclass of tf.keras.Model.

    Functionality:
    #. The class first initializes with default or provided settings.
    #. Then it defines the key components like the transformer encoder layers and output layer.
    #. In the call function, it takes input and passes it through each transformer layer followed by normalization and dense layer for final output.

    Args:
        kwargs (dict): Any additional arguments. If not provided, defaults will be used.

    :param kwargs: Any additional arguments.
    :type kwargs: dict

    :returns: None
    :rtype: None

    .. note::
        The implementation is based on the "Attention is All You Need" paper.

    .. warning::
        The inputs should have a shape of (batch_size, seq_length, height, width), otherwise, a ValueError will be raised.
    """
    DEFAULTS = dict(
        d_model=256,
        n_head=8,
        dim_feedforward=512,
        dropout=0.1,
        num_layers=2,
        num_classes=N_CLASSES,
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}

        # Transformer layers
        self.transformer = [
            TransformerEncoderLayer(
                d_model=self.settings['d_model'],
                n_head=self.settings['n_head'],
                dim_feedforward=self.settings['dim_feedforward'],
                dropout=self.settings['dropout'],
            )
            for _ in range(self.settings['num_layers'])
        ]

        # Normalization layer
        self.norm = LayerNormalization(epsilon=1e-5)

        # Output layer
        self.output_layer = Dense(self.settings['num_classes'])

    @tf.function
    def call(self, inputs, training=False):
        # Check input shape
        if len(inputs.shape) != 4:
            raise ValueError(f'Expected input of shape (batch_size, seq_length, height, width), got {inputs.shape}')

        # Flatten the last two dimensions
        batch_size, seq_length, height, width = inputs.shape
        inputs = tf.reshape(inputs, [batch_size, seq_length, height * width])

        # Pass the input sequence through the Transformer layers
        x = inputs
        for layer in self.transformer:
            x = layer(x, training=training)

        # Apply normalization
        x = self.norm(x)

        # Take the mean of the transformed sequence over the time dimension
        pooled = tf.reduce_mean(x, axis=1)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output

    def compile(self, **kwargs):
        super(TransformerSequenceClassifier, self).compile(**kwargs)


class TransformerPredictor(BaseModel):

    """
    ===================
    TransformerPredictor
    ===================
    A Transformer Predictor model that extends the BaseModel.

    Functionality:
    #. The class first initializes with the learning rate and other parameters.
    #. It then creates an instance of TransformerSequenceClassifier.
    #. It also sets up the learning rate scheduler and the optimizer.
    #. In the call function, it simply runs the TransformerSequenceClassifier.

    Args:
        kwargs (dict): A dictionary of arguments.

    :param kwargs: A dictionary of arguments.
    :type kwargs: dict

    :returns: None
    :rtype: None

    .. note::
        The learning rate is set up with an exponential decay schedule.

    .. warning::
        The learning rate and gamma for the decay schedule must be specified in the 'kwargs'.
    """
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the Transformer model
        self.model = TransformerSequenceClassifier(**kwargs)

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=kwargs["gamma"]
        )

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.scheduler)

        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=[self.accuracy])
        self.compile()

    def call(self, inputs, training=True):
        return self.model(inputs, training=self.training)


class LSTMClassifier(Model):
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

        # LSTM layer
        self.lstm = LSTM(
            self.settings['hidden_dim'],
            return_sequences=True,
            return_state=True,
            dropout=self.settings['dropout'],
            recurrent_dropout=self.settings['dropout'],
            stateful=False
        )

        # Dropout layer
        self.dropout = Dropout(self.settings['dropout'])

        # Readout layer
        self.output_layer = Dense(self.settings['output_dim'])

    @tf.function
    def call(self, inputs):
        """Forward pass through the model"""
        # Adjust the input shape
        # Flatten the last two dimensions
        batch_size, seq_length, height, width = inputs.shape
        x = tf.reshape(inputs, [batch_size, seq_length, -1])

        # LSTM forward pass
        out, _, _ = self.lstm(x)

        # Apply dropout
        out = self.dropout(out)

        # Index hidden state of last time step
        out = self.output_layer(out[:, -1, :])

        return out


class LSTMPredictor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"])

        self.learning_rate = kwargs["learning_rate"]

        # Instantiate the LSTM model
        self.model = LSTMClassifier(**kwargs)

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)


        self.compile(
            optimizer=self.optimizer,
            loss=self.criterion,
            metrics=[self.accuracy])

    @tf.function
    def call(self, inputs):
        return self.model(inputs)


class HybridModel(BaseModel):
    def __init__(self, **kwargs):
        common_params = kwargs['common_params']
        transformer_kwargs = kwargs['transformer_params']
        lstm_kwargs = kwargs['lstm_params']

        super().__init__(learning_rate=common_params["learning_rate"])

        self.lstm = LSTMClassifier(**lstm_kwargs)
        self.transformer = TransformerSequenceClassifier(**transformer_kwargs)
        self.fc = Dense(common_params["num_classes"])

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)

    @tf.function
    def call(self, inputs, training=True):
        lstm_output = self.lstm(inputs, training=training)
        transformer_output = self.transformer(inputs, training=training)

        # Concatenate the outputs of LSTM and Transformer along the feature dimension
        combined = tf.concat([lstm_output, transformer_output], axis=-1)

        # Pass the combined output through the final fully-connected layer
        output = self.fc(combined)

        return output

class CVTransferLearningModel(BaseModel):
    DEFAULTS = dict({})

    def __init__(self, **kwargs):

        # Override defaults with passed-in values
        self.settings = {**self.DEFAULTS, **kwargs}
        super().__init__(learning_rate=self.settings['hparams']['learning_rate'])

        # get weights
        if "weights" not in self.settings['hparams'].keys():
            self.settings['hparams']['weights'] = None



        model = tf.keras.applications.resnet.ResNet152(
            include_top=True,
            weights=self.settings['hparams']['weights'],
            input_tensor=None,
            input_shape=(64,48,3),
            pooling=None,
            classes=self.settings['params']['num_classes'],
        )




        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=self.settings['hparams']['gamma']
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)

        self.model = model



    @tf.function
    def call(self, inputs, training=True):

        # Reshape the array to (-1, 64, 48, 3)
        reshaped_array = tf.reshape(inputs, (-1, 64, 48, 2))

        # Pad the array
        padded_array = tf.concat([reshaped_array,tf.zeros((inputs.shape[0],64,48,1))],axis=-1)

        output = self.model(padded_array)

        return output
