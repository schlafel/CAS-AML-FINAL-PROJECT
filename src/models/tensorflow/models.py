import sys

sys.path.insert(0, "./..")

from config import DEVICE, N_CLASSES

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self, learning_rate):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        self.optimizer = None
        self.scheduler = None

        self.training = False

        self.compile()

    def calculate_accuracy(self, y_pred, y_true):
        preds = tf.reshape(tf.argmax(y_pred, axis=1),[-1, 1])
        targets = tf.reshape(y_true, [-1, 1])
        return self.accuracy(targets, preds)

    def call(self, inputs, training=False):
        raise NotImplementedError()

    def training_step(self, batch):
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

        del landmarks, labels

        return loss.numpy(), accuracy.numpy()

    def validation_step(self, batch):
        landmarks, labels = batch

        predictions = self(landmarks, training=False)

        loss = 0
        if labels is not None:
            loss = self.criterion(labels, predictions)

        accuracy = self.calculate_accuracy(predictions, labels)

        return loss.numpy(), accuracy.numpy()

    def test_step(self, batch):

        landmarks, labels = batch

        predictions = self(landmarks, training=self.training)

        loss = 0
        if labels is not None:
            loss = self.criterion(labels, predictions)

        preds = tf.argmax(predictions, axis=-1)
        accuracy = self.calculate_accuracy(predictions, labels)

        return loss.numpy(), accuracy.numpy(), preds.numpy()

    def optimize(self):
        self.training = True

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def step_scheduler(self):
        pass

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads=n_head, key_dim=d_model)
        self.feed_forward = tf.keras.Sequential([
            Dense(dim_feedforward, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.self_attn(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        feedforward_output = self.feed_forward(out1)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        out2 = self.layernorm2(out1 + feedforward_output)

        return out2

class TransformerSequenceClassifier(Model):
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

        # Output layer
        self.output_layer = Dense(self.settings['num_classes'])

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

        # Take the mean of the transformed sequence over the time dimension
        pooled = tf.reduce_mean(x, axis=1)

        # Pass the pooled sequence through the output layer
        output = self.output_layer(pooled)

        return output

class TransformerPredictor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=kwargs["learning_rate"])

        self.learning_rate=kwargs["learning_rate"]

        # Instantiate the Transformer model
        self.model = TransformerSequenceClassifier(**kwargs)

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.scheduler)

        self.compile()

    def call(self, inputs, training=True):
        return self.model(inputs, training=self.training)
