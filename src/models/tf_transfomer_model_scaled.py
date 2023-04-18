import tensorflow as tf
from src.data.data_utils import get_stratified_TrainValFrames, load_relevant_data_subset
import numpy as np
from src.config import *
import os
from tqdm import tqdm
from src.models.preprocessing_layers import PreprocessLayer

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
    def random_scaling(frames, scale_range=(0.9, 1.1)):
        """
        Apply random scaling to landmark coordinates.

        Args:
            frames (numpy.ndarray): An array of landmarks data.
            scale_range (tuple): A tuple containing the minimum and maximum scaling factors (default: (0.9, 1.1)).

        Returns:
            numpy.ndarray: An array of landmarks with randomly scaled coordinates.
        """
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return frames.numpy() * scale_factor

    def random_rotation(frames, max_angle=10):
        """
        Apply random rotation to landmark coordinates.

        Args:
            frames (numpy.ndarray): An array of landmarks data.
            max_angle (int): The maximum rotation angle in degrees (default: 10).

        Returns:
            numpy.ndarray: An array of landmarks with randomly rotated coordinates.
        """
        # Define Rotation Matrix
        angle = np.radians(np.random.uniform(-max_angle, max_angle))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        if frames.shape[-1] == 3:
            arr_rot_2d = np.einsum('ijk,kl->ijl', frames.numpy()[:, :, 0:2], rotation_matrix)
            c = frames.numpy()[:, :, 2][..., np.newaxis]
            arr_rot = np.concatenate([arr_rot_2d, c], axis=2)

        else:
            arr_rot = np.einsum('ijk,kl->ijl', frames.numpy(), rotation_matrix)

        return arr_rot

    def mirror_landmarks(frames):
        """
        Invert/mirror landmark coordinates along the x-axis.

        Args:
            frames (numpy.ndarray): An array of landmarks data.

        Returns:
            numpy.ndarray: An array of inverted landmarks.
        """
        inverted_frames = np.copy(frames.numpy())
        inverted_frames[:, :, 0] = -inverted_frames[:, :, 0] + 1
        return inverted_frames

    def frame_dropout(frames, dropout_rate=0.05):
        """
        Randomly drop frames from the input landmark data.

        Args:
            frames (numpy.ndarray): An array of landmarks data.
            dropout_rate (float): The proportion of frames to drop (default: 0.05).

        Returns:
            numpy.ndarray: An array of landmarks with dropped frames.
        """
        keep_rate = 1 - dropout_rate
        keep_indices = np.random.choice(len(frames), int(len(frames) * keep_rate), replace=False)
        keep_indices = np.sort(keep_indices)
        dropped_landmarks = frames.numpy()[keep_indices]
        return dropped_landmarks

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
    # Preprocess the data

    prepare_data_Masking(X_train, preprocess_layer, flag="train")
    prepare_data_Masking(X_val, preprocess_layer, flag="val")

    # create dataset
    train_data = get_dataloader(
        flag="train",
        augment_data=True,
        batch_size=32)

    for (X, idx), y in train_data:
        print(X.shape, idx.shape, y.shape)

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
        dataset = dataset.map(augment)

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
    run_transformer()
