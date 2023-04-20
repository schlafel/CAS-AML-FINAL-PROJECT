"""
Configuration
"""
#: This file keeps the project configurations
import os
import torch
import numpy as np

#: Set the device for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
#: Root directory
ROOT_PATH = os.path.join(os.path.dirname(__file__), '../')

#: Directory paths
#: Source files Directory path
SRC_DIR = 'src/'
#: Data files Directory path
DATA_DIR = 'data/'
#: Raw Data files Directory path
RAW_DATA_DIR = 'data/raw/'
#: Processed Data files Directory path
PROCESSED_DATA_DIR = 'data/processed/'
#: Model files Directory path
MODEL_DIR = 'models/'
#: Checkpoint files Directory path
CHECKPOINT_DIR = 'checkpoints/'
#: Run files Directory path
RUNS_DIR = 'runs/'
#: Output files Directory path
OUT_DIR = 'out/'

#: Set Random Seed
SEED = 0

#: Set seed for reproducibility
torch.manual_seed(SEED)

#: Training hyperparameters
#: Training Batch Size
BATCH_SIZE = 128
#: Training Learning rate
LEARNING_RATE = 0.001
#: Training Number of epochs
EPOCHS = 10
#: Training Train set split size
TRAIN_SIZE = 0.85
#: Training Validation set size
VALID_SIZE = 0.05
#: Testing Test set size
TEST_SIZE = 0.1

#: Data files
TRAIN_CSV_FILE = 'train.csv'
MAP_JSON_FILE = 'sign_to_prediction_index_map.json'
LANDMARK_FILES = 'train_landmark_files'
MARKER_FILE = 'preprocessed_data.marker'
CLEANED_FILE = 'cleansed_data.marker'

# Data features
ROWS_PER_FRAME = 543
FACE_FEATURES = 468
HAND_FEATURES = 21
POSE_FEATURES = 33

#: Columns to use
COLUMNS_TO_USE = ["x", "y"]

# Landmarks to select
FACE_FEATURE_START = 0
POSE_FEATURE_START = 489
LEFT_HAND_FEATURE_START = 468
RIGHT_HAND_FEATURE_START = 522

#: Landmarks for Lips
LIPS_LANDMARKS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])

FACE_LANDMARKS = np.array(
    [7, 10, 21, 33, 46, 52, 53, 54, 55, 58, 63, 65, 66, 67, 70, 93, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145,
     148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 234, 246, 249, 251, 263, 276,
     282, 283, 284, 285, 288, 293, 295, 296, 297, 300, 323, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 377, 378,
     379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 454, 466])
#: Landmarks for face
USEFUL_FACE_LANDMARKS = np.array(LIPS_LANDMARKS.tolist() + FACE_LANDMARKS.tolist())
#: Landmarks for pose
USEFUL_POSE_LANDMARKS = np.arange(500, 514)
#: Landmarks for left hand
USEFUL_LEFT_HAND_LANDMARKS = np.arange(468, 489)
#: Landmarks for right hand
USEFUL_RIGHT_HAND_LANDMARKS = np.arange(522, 543)
#: Landmarks for both hands
USEFUL_HAND_LANDMARKS = np.concatenate((USEFUL_LEFT_HAND_LANDMARKS, USEFUL_RIGHT_HAND_LANDMARKS), axis=0)
#: All Landmarks 
USEFUL_ALL_LANDMARKS = np.concatenate((USEFUL_FACE_LANDMARKS, USEFUL_HAND_LANDMARKS, USEFUL_POSE_LANDMARKS))

USED_FACE_FEATURES = len(USEFUL_FACE_LANDMARKS)
USED_POSE_FEATURES = len(USEFUL_POSE_LANDMARKS)
USED_HAND_FEATURES = len(USEFUL_LEFT_HAND_LANDMARKS)

N_CLASSES = 250
N_LANDMARKS = USED_FACE_FEATURES + 2 * USED_HAND_FEATURES + USED_POSE_FEATURES
N_DIMS = len(COLUMNS_TO_USE)

# List with landmarks to use
FACE_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_FACE_LANDMARKS)).squeeze()
LIPS_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, LIPS_LANDMARKS)).squeeze()
POSE_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_POSE_LANDMARKS)).squeeze()
HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_HAND_LANDMARKS)).squeeze()
LEFT_HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_LEFT_HAND_LANDMARKS)).squeeze()
RIGHT_HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_RIGHT_HAND_LANDMARKS)).squeeze()

INTEREMOLATE_MISSING = 3
SKIP_CONSECUTIVE_ZEROS = 4
INPUT_SIZE = 32
MIN_SEQUEENCES = INPUT_SIZE / 4
