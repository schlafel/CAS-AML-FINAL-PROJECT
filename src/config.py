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