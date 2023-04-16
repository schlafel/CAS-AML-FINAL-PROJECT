"""
Configuration
"""
#: This file keeps the project configurations

import torch

#: Set the device for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
#: Root directory
ROOT_PATH='../'

#: Directory paths
#: Source files Directory path
SRC_DIR='src/'
#: Data files Directory path
DATA_DIR='data/'
#: Raw Data files Directory path
RAW_DATA_DIR='data/raw/'
#: Processed Data files Directory path
PROCESSED_DATA_DIR='data/processed/'
#: Model files Directory path
MODEL_DIR='models/'
#: Checkpoint files Directory path
CHECKPOINT_DIR = 'checkpoints/'
#: Run files Directory path
RUNS_DIR='runs/'
#: Output files Directory path
OUT_DIR='out/'

#: Set Random Seed
SEED=0

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
TRAIN_SIZE=0.85
#: Training Validation set size
VALID_SIZE=0.05
#: Testing Test set size
TEST_SIZE=0.1

#: Data files
TRAIN_CSV_FILE = 'train.csv'
MAP_JSON_FILE = 'sign_to_prediction_index_map.json'
LANDMARK_FILES = 'train_landmark_files'
MARKER_FILE = 'preprocessed_data.marker'
CLEANED_FILE = 'cleansed_data.marker'

# Data features
FACE_FEATURES=468
HAND_FEATURES=21
POSE_FEATURES=33
MAX_SEQUENCES=150
MIN_SEQUENCES=15

# Landmarks to select
#: Landmarks for face
USEFUL_FACE_LANDMARKS = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
#: Landmarks for pose
USEFUL_POSE_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

USED_FACE_FEATURES=len(USEFUL_FACE_LANDMARKS)
USED_POSE_FEATURES=len(USEFUL_POSE_LANDMARKS)
USED_HAND_FEATURES=21

#List with landmarks to use
LEFT_HAND_INDICES  = list(range(USED_FACE_FEATURES,USED_FACE_FEATURES+USED_HAND_FEATURES))
RIGHT_HAND_INDICES = list(range(USED_FACE_FEATURES+USED_HAND_FEATURES+POSE_FEATURES,USED_FACE_FEATURES+USED_HAND_FEATURES+POSE_FEATURES+USED_HAND_FEATURES))
POSE_INDICES       = list(range(USED_FACE_FEATURES+USED_HAND_FEATURES,USED_FACE_FEATURES+USED_HAND_FEATURES+USED_POSE_FEATURES))

N_CLASSES = 250
N_LANDMARKS = USED_FACE_FEATURES+2*USED_HAND_FEATURES+USED_POSE_FEATURES

SKIP_CONSECUTIVE_ZEROS=4

#: Columns to use
COLUMNS_TO_USE = ["x","y"]