import tensorflow as tf
from src.data.data_utils import get_stratified_TrainValFrames, load_relevant_data_subset
from src.augmentations import *
import os
from tqdm import tqdm
from src._old.preprocessing_layers import PreprocessLayer

# Model Config
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
        POSE_IDX  # Pose
)

LIPS_IDX_PROC = np.argwhere(np.isin(useful_landmarks, LIPS_IDX)).squeeze()
LHAND_IDX_PROC = np.argwhere(np.isin(useful_landmarks, USEFUL_LEFT_HAND_LANDMARKS)).squeeze()
RHAND_IDX_PROC = np.argwhere(np.isin(useful_landmarks, USEFUL_RIGHT_HAND_LANDMARKS)).squeeze()
POSE_IDX_PROC = np.argwhere(np.isin(useful_landmarks, POSE_IDX)).squeeze()

HAND_IDX_PROC = LHAND_IDX_PROC + RHAND_IDX_PROC
print(len(useful_landmarks))



def augment(x, y, augmentation_threshold=.5):
    x, idx = x
    if tf.random.uniform([]) > augmentation_threshold:
        [x, ] = tf.py_function(random_scaling, [x], [tf.float32])
    if tf.random.uniform([]) > augmentation_threshold:
        [x, ] = tf.py_function(random_rotation, [x], [tf.float32])
    if tf.random.uniform([]) > augmentation_threshold:
        [x, ] = tf.py_function(mirror_landmarks, [x], [tf.float32])
    # if tf.random.uniform([]) > augmentation_threshold:
    #     [x, ] = tf.py_function(frame_dropout, [x], [tf.float32])

    return (x, idx), y




def run_transformer():
    X_train, X_val = get_stratified_TrainValFrames()
    print(X_train.shape, X_val.shape)

    # Get Mean values
    mn = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "statistics", "mean_92.npy"))
    std = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "statistics", "std_92.npy"))

    LIPS_MEAN = mn[LIPS_IDX_PROC, :]
    LIPS_STD = std[LIPS_IDX_PROC, :]

    LH_MEAN = mn[LHAND_IDX_PROC, :]
    LH_STD = std[LHAND_IDX_PROC, :]

    RH_MEAN = mn[RHAND_IDX_PROC, :]
    RH_STD = std[RHAND_IDX_PROC, :]

    POSE_MEAN = mn[POSE_IDX_PROC, :]
    POSE_STD = std[POSE_IDX_PROC, :]

    print("done")
    ######### Get the data
    # Initialize Preprocess Layer
    preprocess_layer = PreprocessLayer(seq_length=32,
                                       useful_landmarks=useful_landmarks,
                                       hand_idxs=(USEFUL_LEFT_HAND_LANDMARKS.tolist() +
                                                  USEFUL_RIGHT_HAND_LANDMARKS.tolist()))
    ######### Preprocess the data #########
    if PREPROCESS:
        prepare_data_Masking(X_train, preprocess_layer, flag="train")
        prepare_data_Masking(X_val, preprocess_layer, flag="val")

    ######### GET DATA #########
    train_data = get_dataloader(
        flag="train",
        augment_data=True,
        batch_size=BATCH_SIZE)

    val_data = get_dataloader(
        flag="validation",
        augment_data=False,
        batch_size=BATCH_SIZE)


    ######### DEFINE MODEL #########

    # for (X, idx), y in train_data:
    #     print(X.shape, idx.shape, y.shape)

    # val_data = prepare_data_Masking(X_val, preprocess_layer)


def get_dataloader(flag="train",
                   augment_data=True,
                   batch_size=250):
    X = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"X_{flag}.npy"))
    tgt = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"y_{flag}.npy"))
    idx_0 = np.load(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"idx_{flag}.npy"))

    dataset = tf.data.Dataset.from_tensor_slices(
        ((X, idx_0),tgt)
    )

    dataset = dataset.shuffle(len(idx_0))
    if augment_data:
        dataset = dataset.map(augment,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache()

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def prepare_data_Masking(df,
                         preprocess_layer,
                         flag,
                         ):
    """

    :param df:
    :param preprocess_layer:
    :param flag:
    :return:
    """

    # save the preprocessing csv to the file...
    df.to_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"{flag}.csv"))

    X = []
    idx_0 = []
    tgt = []

    for _i, row in tqdm(df.iterrows(), total=len(df)):
        arr_in = load_relevant_data_subset(row.file_path)
        x_prep, idx = preprocess_layer(arr_in)
        X.append(x_prep)
        idx_0.append(idx)
        tgt.append([row.target])

    np.save(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"X_{flag}.npy"), np.array(X))
    np.save(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"y_{flag}.npy"), np.array(tgt))
    np.save(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "train_landmark_92", f"idx_{flag}.npy"), np.array(idx_0))

    print("Successfully saved the preprocessed files")


if __name__ == '__main__':
    PREPROCESS = False
    BATCH_SIZE = 256
    run_transformer()
