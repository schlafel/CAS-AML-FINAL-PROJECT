# This file keeps the project configurations

import torch
import random

# Set the device for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
ROOT_PATH='../'

SRC_DIR='src/'
DATA_DIR='data/'
RAW_DATA_DIR='data/raw/'
PROCESSED_DATA_DIR='data/processed/'
MODEL_DIR='models/'
CHECKPOINT_DIR = 'checkpoints/'
RUNS_DIR='runs/'
OUT_DIR='out/'

# Set Random Seed
SEED=0

# Set seed for reproducibility
torch.manual_seed(SEED)

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Data directories
TRAIN_CSV_FILE = 'train.csv'
MAP_JSON_FILE = 'sign_to_prediction_index_map.json'
LANDMARK_FILES = 'train_landmark_files'
MARKER_FILE = 'preprocessed_data.marker'

# Data features
FACE_FEATURES=468
HAND_FEATURES=21
POSE_FEATURES=33
MAX_SEQUENCES=173

# Landmarks to select
USEFUL_FACE_LANDMARKS = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
USEFUL_POSE_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

USED_FACE_FEATURES=len(USEFUL_FACE_LANDMARKS)
USED_POSE_FEATURES=len(USEFUL_POSE_LANDMARKS)
USED_HAND_FEATURES=21

N_CLASSES = 250
N_LANDMARKS = USED_FACE_FEATURES+2*USED_HAND_FEATURES+USED_POSE_FEATURES

SKIP_CONSECUTIVE_ZEROS=4

COLUMNS_TO_USE = ["x","y"]

#List with landmarks to use
LEFT_HAND_INDICES = list(range(FACE_FEATURES,FACE_FEATURES+HAND_FEATURES))
RIGHT_HAND_INDICES = list(range(FACE_FEATURES+HAND_FEATURES+POSE_FEATURES,FACE_FEATURES+HAND_FEATURES))
LIPS_INDICES = lip_landmarks = [
61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
FACE_INDICES = [] + LIPS_INDICES
POSE_INDICES = list(range(FACE_FEATURES  + HAND_FEATURES  + 11 ,FACE_FEATURES + HAND_FEATURES + 23))
LANDMARK_INDICES = LEFT_HAND_INDICES + RIGHT_HAND_INDICES + FACE_INDICES + POSE_INDICES