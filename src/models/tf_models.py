import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from src.models.utils import scaled_dot_product


#import pydevd



class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.dense_1 = layers.Dense(dff, activation='relu')
        self.dense_2 = layers.Dense(d_model, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)

        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, inputs,training = False,**kwargs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dropout(x)

        x = self.add([inputs, x])
        x = self.layer_norm(x)
        return x

@tf.function
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

    @tf.function
    def call(self, x,training = False, **kwargs):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                                      key_dim = d_model,
                                                      value_dim=d_model
                                                      )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        self.dropout = layers.Dropout(rate=dropout_rate)

    @tf.function
    def call(self, x, training = False, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        #tf.print(x.shape)
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

    @tf.function
    def call(self, inputs, training = False, **kwargs):
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

    @tf.function
    def call(self, x, training = False, **kwargs):
        for enc in self.enc:
            x = enc(x,training = training)
        return x

class TransformerClassifierModel(Model):
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


    @tf.function
    def call(self, inputs, training=False, **kwargs):
        x = self.pos_encoding(inputs)
        x = self.encoder(x,training=training)
        x = self.avgPool(x)
        #pydevd.settrace(suspend=False)
#        tf.print(x.shape,type(x))
        return self.fc(x)

    def compile(self, **kwargs):
        super(TransformerClassifierModel, self).compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],

            **kwargs
        )


class LSTM_BASELINE_TF(Model):
    def __init__(self,
                 input_shape = (150,184,2),
                 n_hidden = 256,
                 dropout = .25,
                 n_LSTM_LAYERS = 3,
                 num_classes = 250):
        super().__init__()
        self.inp_shp = input_shape
        # self.reshp_1 = tf.keras.layers.Reshape((input_shape[0],input_shape[1]*input_shape[2]),
        #                                   input_shape=input_shape,
        #                                         )
        self.shape_lstm = (input_shape[0],input_shape[1]*input_shape[2])

        self.hidden_lstm = [tf.keras.layers.LSTM(n_hidden,
                                                 return_sequences = True,
                                                 name = f"HIDDEN_LSTM_{i}") for i in range(n_LSTM_LAYERS-1)]
        self.dropouts = [tf.keras.layers.Dropout(dropout) for _ in range(n_LSTM_LAYERS-1)]

        self.final_lstm = tf.keras.layers.LSTM(n_hidden,return_sequences = False,
                                               name = "Final_LSTM_LAYER")

        self.fc_layer = tf.keras.layers.Dense(num_classes,activation = "softmax")

        # self.model()

    @tf.function
    def call(self,inputs,training = False):
        #Reshape the inputs
        # x = self.reshp_1(inputs)
        x = tf.reshape(inputs,shape=[-1, *self.shape_lstm])
        #Run through the hidden_lstm
        for lstm,dropout in zip(self.hidden_lstm,self.dropouts):
            x = lstm(x)
            x = dropout(x)


        #do the final lstm
        x = self.final_lstm(x)

        #Finally the fc-Layer
        out = self.fc_layer(x)
        return out

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(self.inp_shp),dtype = tf.float64)
    #     return Model(inputs=[x], outputs=self.call(x))
    #

class Very_simple_model(tf.keras.models.Model):
    def __init__(self,shapes = (150,184*2)):
        super().__init__()
        self.flat = tf.keras.layers.Flatten()

        self.reshape = tf.keras.layers.Reshape((150,184*2))
        self.dl = [tf.keras.layers.Dense(100) for _ in range(3)]
        self.otpt = tf.keras.layers.Dense(250,activation = "softmax")
        self.mdl_shp = shapes
    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, shape=[-1, *self.mdl_shp])
        for lay in self.dl:
            x = lay(x)
        x = self.flat(inputs)

        return self.otpt(x)



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


# Full Transformer
class Transformer(tf.keras.Model):
    def __init__(self,
                 num_blocks,
                 mha_units = 384,
                 mlp_ratio = 2,
                 layer_norm_epsilon = 1e-6,
                 mlp_dropout = 0.3,
                 tansformer_heads = 8):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks

        self.mha_units = mha_units #embedding size
        self.mlp_ratio = mlp_ratio
        self.layer_norm_epsilon = layer_norm_epsilon #embedding size
        self.mlp_dropout = mlp_dropout #embedding size
        self.tansformer_heads = tansformer_heads #embedding size



    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.mha_units, self.tansformer_heads))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.mha_units * self.mlp_ratio,
                                      activation=tf.keras.activations.gelu,
                                      kernel_initializer=tf.keras.initializers.glorot_uniform),
                tf.keras.layers.Dropout(self.mlp_dropout),
                tf.keras.layers.Dense(self.mha_units,
                                      kernel_initializer=tf.keras.initializers.he_uniform),
            ]))

    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2

        return x


class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units

    def build(self, input_shape):
        # Initiailizers
        INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
        INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
        INIT_ZEROS = tf.keras.initializers.constant(0.0)


        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=True,
                                  kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=True,
                                  kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )


class Embedding(tf.keras.Model):
    def __init__(self, lips_units= 384,
                 hands_units= 384,
                 pose_units= 384,
                 units = 384):
        super(Embedding, self,).__init__()

        self.lips_units = lips_units
        self.hands_units = hands_units
        self.pose_units = pose_units
        self.units = units

    def get_diffs(self, l):
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0, 1, 3, 2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, INPUT_SIZE, S * S])
        return diffs

    def build(self, input_shape):
        INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
        INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
        INIT_ZEROS = tf.keras.initializers.constant(0.0)


        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(INPUT_SIZE + 1, self.units, embeddings_initializer=INIT_ZEROS)
        self.positional_weight = tf.Variable(1.)
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(self.lips_units, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(self.hands_units, 'left_hand')
        self.right_hand_embedding = LandmarkEmbedding(self.hands_units, 'right_hand')
        self.pose_embedding = LandmarkEmbedding(self.pose_units, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name='fully_connected_1', use_bias=False,
                                  kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(self.units, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name='fc')

    def call(self, lips0, left_hand0, right_hand0, pose0, non_empty_frame_idxs, training=False):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Right Hand
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3)
        # Merge Landmarks with trainable attention weights
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_weight * self.positional_embedding(normalised_non_empty_frame_idxs)

        return x



if __name__ == '__main__':
    # # test the encoding layer
    # pos_enc = PositionalEncoding(d_model=192, seq_length=150)
    # x_in = np.random.random((1,150, 192))
    # pos_enc(x_in)
    # print("done")
    #
    #
    # #test the encoder
    # simpl_enc = Encoder(
    #     d_model = 192,
    #         num_heads=8,
    #         )
    # enc_out = simpl_enc(x_in)
    # print(enc_out.shape)
    #
    # ####### Test the Transformer Architecture #######
    # input_shape = (150,152)
    #
    # model = TransformerClassifierModel(
    #     num_classes=250,
    #     input_shape=input_shape,
    #     num_heads=10,
    #     dff=128,
    #     num_encoder_blocks=5,
    #     dropout_rate=0.1,
    # )
    # # generate some data
    #
    # model.build(input_shape=input_shape)
    # model.compile()
    #
    #
    # X_in = np.random.random((64,*input_shape))
    # mod_out = model(X_in)
    # print(X_in.shape,mod_out.shape)
    # #model.build((64,150,128))
    #
    #
    # print(model.summary())
    #

    #Test the models
    from src.config import  *
    from src.data.dataset import ASL_DATASET_TF
    dataset_cls = ASL_DATASET_TF()
    dataset = dataset_cls.create_dataset(batch_size = 250)

    csv_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE)
    data_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR)




    #
    for batchX,batchY in dataset:
        break

    print(batchX.shape,batchX.dtype)

    model = LSTM_BASELINE_TF()
    model = Very_simple_model()

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Reshape((150, 184*2), input_shape=(150,184,2)),
    #     tf.keras.layers.LSTM(64,return_sequences = True),
    #     tf.keras.layers.LSTM(128,return_sequences = True),
    #     tf.keras.layers.LSTM(256),
    #     tf.keras.layers.Dense(75, activation='relu'),
    #     tf.keras.layers.Dense(250, activation='softmax')
    # ])

    model.build(input_shape=(None, 32, 184, 2))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary(expand_nested=True))

    model.fit(dataset,
              epochs = 150,
              )



