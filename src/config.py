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
MAX_SEQUENCES=537
N_CLASSES = 250

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