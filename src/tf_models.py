import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Dense, \
    Dropout
import numpy as np


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=seq_length, depth=d_model)

    def call(self, x, *args, **kwargs):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dff: int = 1024,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                      key_dim = d_model,
                                                      value_dim=d_model
                                                      )
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dff: int = 1024,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.self_att_block = SelfAttentionBlock(d_model = d_model,num_heads = num_heads)
        self.ff_block = FeedForward(d_model=d_model,dff=dff,dropout_rate=dropout_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.self_att_block(inputs)
        x = self.ff_block(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_encoder_blocks: int = 1,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dff: int = 1024,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.enc = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for n in range(n_encoder_blocks)]

    def call(self, x, *args, **kwargs):
        for enc in self.enc:
            x = enc(x)
        return x


class TransformerClassifierModel(tf.keras.Model):
    def __init__(self,
                 input_shape=(150, 192),
                 num_classes: int = 250,
                 num_encoder_blocks: int = 1,
                 num_heads: int = 8,
                 dff=1024,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.seq_length = input_shape[0]
        self.d_model = input_shape[1]
        self.pos_encoding = PositionalEncoding(*input_shape)

        self.encoder = Encoder(n_encoder_blocks=num_encoder_blocks,
                               d_model=self.d_model,
                               num_heads=num_heads,
                               dff=dff,
                               dropout_rate=dropout_rate)
        self.avgPool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        # Overwrite summary method to return accurate summary statistics...
        x = Input(shape=(self.seq_length,self.d_model))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def call(self, inputs, training=None, mask=None):
        x = self.pos_encoding(inputs)
        x = self.encoder(x)
        x = self.avgPool(x)
        return self.fc(x)

    def compile(self, **kwargs):
        super(TransformerClassifierModel, self).compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
            **kwargs
        )


if __name__ == '__main__':
    # test the encoding layer
    pos_enc = PositionalEncoding(d_model=192, seq_length=150)
    x_in = np.random.random((1,150, 192))
    pos_enc(x_in)
    print("done")


    #test the encoder
    simpl_enc = Encoder(
        d_model = 192,
            num_heads=8,
            )
    enc_out = simpl_enc(x_in)
    print(enc_out.shape)

    ####### Test the Transformer Architecture #######
    input_shape = (150,152)

    model = TransformerClassifierModel(
        num_classes=250,
        input_shape=input_shape,
        num_heads=10,
        dff=128,
        num_encoder_blocks=5,
        dropout_rate=0.1,
    )
    # generate some data

    model.build(input_shape=input_shape)
    model.compile()
    print(model.summary())


    X_in = np.random.random((64,*input_shape))
    mod_out = model(X_in)
    print(X_in.shape,mod_out.shape)
    #model.build((64,150,128))

