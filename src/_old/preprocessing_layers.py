import tensorflow as tf
from src.config import *
from src.data.data_utils import load_relevant_data_subset
from tqdm import tqdm
import pyarrow.parquet as pq

"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
"""


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self,
                 seq_length=32,
                 useful_landmarks = USEFUL_ALL_LANDMARKS.tolist(),
                 hand_idxs = USEFUL_HAND_LANDMARKS.tolist(),
                 ):
        super(PreprocessLayer, self).__init__()
        self.SEQ_LENGTH = seq_length
        self.USEFUL_LANDMARKS_IDX = useful_landmarks
        self.USEFUL_HAND_LANDMARKS = hand_idxs
        self.N_COLS = len(self.USEFUL_LANDMARKS_IDX)
        self.N_DIMS = 3

    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

    #@tf.function(
    #    input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),), )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]

        # Filter Out Frames With Empty Hand Data
        # frames_hands_nansum = tf.experimental.numpy.nanmean(tf.gather(data0, self.HAND_IDX, axis=1), axis=[1, 2])
        data_hands = tf.gather(data0, self.USEFUL_HAND_LANDMARKS, axis=1)
        non_empty_frames_idxs =  tf.squeeze(tf.where(tf.experimental.numpy.nansum(data_hands, axis=(1, 2)) != 0))


        data = tf.gather(data0, non_empty_frames_idxs, axis=0)
        # tf.print(data0.shape, data.shape)


        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]

        # Gather Relevant Landmark Columns
        data = tf.gather(data, self.USEFUL_LANDMARKS_IDX, axis=1)

        # Video fits in INPUT_SIZE
        if N_FRAMES < self.SEQ_LENGTH:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, self.SEQ_LENGTH - N_FRAMES]], constant_values=-1)
            # Pad Data With Zeros
            data = tf.pad(data, [[0, self.SEQ_LENGTH - N_FRAMES], [0, 0], [0, 0]], constant_values=0)
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, tf.cast(non_empty_frames_idxs,tf.float32)
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < self.SEQ_LENGTH ** 2:
                repeats = self.SEQ_LENGTH * self.SEQ_LENGTH // N_FRAMES
                data = tf.repeat(data, repeats=repeats, axis=0)

            pool_size = len(data) // self.SEQ_LENGTH
            if len(data) % self.SEQ_LENGTH > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * self.SEQ_LENGTH) - len(data)
            else:
                pad_size = (pool_size * self.SEQ_LENGTH) % len(data)

            pad_left = pad_size // 2 + self.SEQ_LENGTH // 2
            pad_right = pad_size // 2 + self.SEQ_LENGTH // 2
            if pad_size % 2 > 0:
                pad_right += 1

            data = tf.experimental.numpy.concatenate((tf.repeat(data[:1], repeats=pad_left, axis=0), data),
                                           axis=0)
            data = tf.experimental.numpy.concatenate((data, tf.repeat(data[-1:], repeats=pad_right, axis=0)),
                                           axis=0)

            data = tf.reshape(data,[INPUT_SIZE, -1, N_LANDMARKS, self.N_DIMS])
            data = tf.experimental.numpy.nanmean(data, axis=1)

            size = INPUT_SIZE

            non_empty_frames_idxs = tf.linspace(0, len(USEFUL_ALL_LANDMARKS), INPUT_SIZE)

            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, tf.cast(non_empty_frames_idxs,tf.float32)


# Get complete file path to file
def load_prepare_data(x,y):
    def load_parquet(filename):
        table = pq.read_table(filename.numpy().decode(), columns=["x", "y", "z"])
        data = table.to_pandas().values
        # data = table.to_numpy()
        n_frames = int(len(data) / 543)
        arr_out = data.reshape(n_frames, 543, 3)

        # arr_out, ni = prep_layer(arr_out.astype(np.float32))
        return arr_out.astype(np.float32)





    x = tf.py_function(load_parquet,
                       inp = [x],
                       Tout=tf.float32)
    # tf.print(y)
    return x,[y]

def run_prepLayer(x,y):
    def prep_lyr(x,in_prep_lyr=prep_layer):
        # tf.print(x.numpy().shape)
        x_prep,ind = in_prep_lyr(x.numpy())
        return x_prep,ind
    x,ind= tf.py_function(prep_lyr,
                          inp=[x],

                          Tout=[tf.float32,tf.float32])



    return (x,ind),y

#
# def get_model():
#     # Inputs
#     frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
#     non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
#     # Padding Mask
#     mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
#     mask = tf.expand_dims(mask, axis=2)
#
#     """
#         left_hand: 468:489
#         pose: 489:522
#         right_hand: 522:543
#     """
#     x = frames
#     x = tf.slice(x, [0, 0, 0, 0], [-1, INPUT_SIZE, N_COLS, 2])
#     # LIPS
#     lips = tf.slice(x, [0, 0, LIPS_START, 0], [-1, INPUT_SIZE, 40, 2])
#     lips = tf.where(
#         tf.math.equal(lips, 0.0),
#         0.0,
#         (lips - LIPS_MEAN) / LIPS_STD,
#     )
#     lips = tf.reshape(lips, [-1, INPUT_SIZE, 40 * 2])
#     # LEFT HAND
#     left_hand = tf.slice(x, [0, 0, 40, 0], [-1, INPUT_SIZE, 21, 2])
#     left_hand = tf.where(
#         tf.math.equal(left_hand, 0.0),
#         0.0,
#         (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
#     )
#     left_hand = tf.reshape(left_hand, [-1, INPUT_SIZE, 21 * 2])
#     # RIGHT HAND
#     right_hand = tf.slice(x, [0, 0, 61, 0], [-1, INPUT_SIZE, 21, 2])
#     right_hand = tf.where(
#         tf.math.equal(right_hand, 0.0),
#         0.0,
#         (right_hand - RIGHT_HANDS_MEAN) / RIGHT_HANDS_STD,
#     )
#     right_hand = tf.reshape(right_hand, [-1, INPUT_SIZE, 21 * 2])
#     # POSE
#     pose = tf.slice(x, [0, 0, 82, 0], [-1, INPUT_SIZE, 10, 2])
#     pose = tf.where(
#         tf.math.equal(pose, 0.0),
#         0.0,
#         (pose - POSE_MEAN) / POSE_STD,
#     )
#     pose = tf.reshape(pose, [-1, INPUT_SIZE, 10 * 2])
#
#     x = lips, left_hand, right_hand, pose
#
#     x = Embedding()(lips, left_hand, right_hand, pose, non_empty_frame_idxs)
#
#     # Encoder Transformer Blocks
#     x = Transformer(NUM_BLOCKS)(x, mask)
#
#     # Pooling
#     x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
#     # Classification Layer
#     x = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax,
#                               kernel_initializer=INIT_GLOROT_UNIFORM)(x)
#
#     outputs = x
#
#     # Create Tensorflow Model
#     model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
#
#     # Simple Categorical Crossentropy Loss
#     loss = tf.keras.losses.SparseCategoricalCrossentropy()
#
#     # Adam Optimizer with weight decay
#     optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
#
#     # TopK Metrics
#     metrics = [
#         tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
#         tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
#         tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
#     ]
#
#     model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#
#     return model
#

def get_means(df):
    li_out = []
    for _i,file_path in tqdm(df.file_path.iteritems(),total = len(df)):
        arr_in = load_relevant_data_subset(file_path)
        li_out.append(arr_in[:, useful_landmarks, :2])

    concat_arr = np.concatenate(li_out, axis=0)
    mn = np.nanmean(concat_arr,axis = 0)
    std = np.nanmean(concat_arr,axis = 0)

    return mn,std

if __name__ == '__main__':


    LIPS_IDX = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]

    POSE_IDX = list(range(502, 512))
    useful_landmarks = (
            LIPS_IDX +  # Lips
            USEFUL_LEFT_HAND_LANDMARKS.tolist() +
            USEFUL_RIGHT_HAND_LANDMARKS.tolist() +
            POSE_IDX# Pose
    )

    LIPS_IDX_PROC = np.argwhere(np.isin(useful_landmarks, LIPS_IDX)).squeeze()
    LHAND_IDX_PROC = np.argwhere(np.isin(useful_landmarks, USEFUL_LEFT_HAND_LANDMARKS)).squeeze()
    RHAND_IDX_PROC = np.argwhere(np.isin(useful_landmarks, USEFUL_RIGHT_HAND_LANDMARKS)).squeeze()
    POSE_IDX_PROC = np.argwhere(np.isin(useful_landmarks, POSE_IDX)).squeeze()

    HAND_IDX_PROC = LHAND_IDX_PROC + RHAND_IDX_PROC
    print(len(useful_landmarks))


    df_train = load_train_frame(path_csv=os.path.join(ROOT_PATH,RAW_DATA_DIR,TRAIN_CSV_FILE))
    mn,std = get_means(df_train)

    np.save(f"mean_{len(useful_landmarks)}.npy",mn)
    np.save(f"std_{len(useful_landmarks)}.npy",std)

    LIPS_MEAN = mn[LIPS_IDX_PROC,:]
    LIPS_STD = std[LIPS_IDX_PROC,:]

    LH_MEAN = mn[LHAND_IDX_PROC,:]
    LH_STD = std[LHAND_IDX_PROC,:]


    RH_MEAN = mn[RHAND_IDX_PROC,:]
    RH_STD = std[RHAND_IDX_PROC,:]

    POSE_MEAN = mn[POSE_IDX_PROC,:]
    POSE_STD = std[POSE_IDX_PROC,:]


    print(f"Lips mean,std: {LIPS_MEAN},{LIPS_STD}")
    print(f"Left hand mean,std: {LH_MEAN},{LH_STD}")
    print(f"Right hand mean,std: {RH_MEAN},{RH_STD}")
    print(f"Pose mean,std: {POSE_MEAN},{POSE_STD}")





    prep_layer = PreprocessLayer(seq_length=32,
                                 useful_landmarks=useful_landmarks,
                                 hand_idxs=(USEFUL_LEFT_HAND_LANDMARKS +
                                            USEFUL_RIGHT_HAND_LANDMARKS))


    #get data and run it through prep_layer
    for i in tqdm(range(10)):
        arr = load_relevant_data_subset(df_train.sample(1).file_path.values[0])
        prp_data = prep_layer(arr)

    print("loaded_raw_data")
    # create dataset
    dataset = get_dataset_FROM_RAW(path_train)

    #Run through the dataset
    for ((X,ind),y) in tqdm(dataset):
        break
        # print(X.shape,ind.shape,y.shape)

        # break



    #Model Config
    # Epsilon value for layer normalisation
    LAYER_NORM_EPS = 1e-6

    # Dense layer units for landmarks
    LIPS_UNITS = 384
    HANDS_UNITS = 384
    POSE_UNITS = 384
    # final embedding and transformer embedding size
    UNITS = 384

    # Transformer
    NUM_BLOCKS = 4
    MLP_RATIO = 2

    # Dropout
    EMBEDDING_DROPOUT = 0.00
    MLP_DROPOUT_RATIO = 0.30
    CLASSIFIER_DROPOUT_RATIO = 0.10

    # Initiailizers
    INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
    INIT_ZEROS = tf.keras.initializers.constant(0.0)
    # Activations
    GELU = tf.keras.activations.gelu





