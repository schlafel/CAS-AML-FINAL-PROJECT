"""
=================================
Project configuration description
=================================

This configuration is created allows for easy tuning of your machine learning model's parameters and setup.
The device on which the model runs, the paths for various resources, the seed for random number generation,
hyperparameters for model training, and much more and quickly be change and configured.
This makes your setup flexible and easy to adapt for various experiments and environments
"""
#: This file keeps the project configurations

import os
import torch
import numpy as np
from matplotlib import style

#: Setting the style from matplotlib
style.use(os.path.join(os.path.dirname(__file__),"..","styles","CASAML_Style.mplstyle"))

#
# =============================================================================
# Device Setup
# =============================================================================
#

#: Setting the device for training, 'cuda' if a CUDA-compatible GPU is available,
#: 'mps' if multiple processors are available, 'cpu' if none of the above.
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

#
# =============================================================================
# Paths
# =============================================================================
#

#: Root directory
ROOT_PATH = os.path.join(os.path.dirname(__file__), '../')  # Root directory path

# Directory paths
#: Source files Directory path
SRC_DIR = 'src/'                                            # Source files directory path
#: Data files Directory path
DATA_DIR = 'data/'                                          # Data files directory path
#: Raw Data files Directory path
RAW_DATA_DIR = 'data/raw/'                                  # Raw data files directory path
#: Processed Data files Directory path
PROCESSED_DATA_DIR = 'data/processed/'                      # Processed data files directory path
#: Model files Directory path
MODEL_DIR = 'models/'                                       # Model files directory path
#: Checkpoint files Directory path
CHECKPOINT_DIR = 'checkpoints/'                             # Checkpoint files directory path
#: Run files Directory path
RUNS_DIR = 'runs/'                                          # Run files directory path for tensorboard
#: Output files Directory path
OUT_DIR = 'out/'                                            # Output files directory path

#
# =============================================================================
# Randomization seeds
# =============================================================================
#

#: Set Random Seed
SEED = 0                                                    # Set random seed for reproducibility
#: Set seed for reproducibility
torch.manual_seed(SEED)

#
# =============================================================================
# Training hyperparameters
# =============================================================================
#

#: Tune hyperparameters
TUNE_HP = True                                              # Tune hyperparameters
#: Training Batch Size
BATCH_SIZE = 128                                            # Training batch size
#: Training Learning rate
LEARNING_RATE = 0.001                                       # Training learning rate, may be overridden in model yaml
#: Training Number of epochs
EPOCHS = 50                                                 # Training number of epochs
#: Training Train set split size
TRAIN_SIZE = 0.90                                           # Training set split size
#: Training Validation set size
VALID_SIZE = 0.05                                           # Validation set size
#: Testing Test set size
TEST_SIZE = 0.05                                            # Test set size

#
# =============================================================================
# Early stopping parameters
# =============================================================================
#

#: Which metric should be used for early stopping loss/accuracy
EARLY_STOP_METRIC = "accuracy"                              # Metric used for early stopping
#: What is the mode? min/max
EARLY_STOP_MODE = "max"                                     # Mode for early stopping, can be 'min' or 'max'
#: The number of epochs to wait for improvement in the validation loss before stopping training
EARLY_STOP_PATIENCE = 10                                    # Number of epochs to wait for improvement before stopping
#: The value of loss as margin to tolerate
EARLY_STOP_TOLERENCE = 0.001                                # Tolerance value for loss improvement

#
# =============================================================================
# Dynamic drop out parameters
# =============================================================================
#

#: The value of initial low dropouts rate
DYNAMIC_DROP_OUT_INIT_RATE = 0.01                           # Initial dropout rate foy dynamic dropout
#: The value to increase dropouts by
DYNAMIC_DROP_OUT_REDUCTION_RATE = 1.1                       # Dropout rate reduction factor
#: The max value of dynamic dropouts
DYNAMIC_DROP_OUT_MAX_THRESHOLD = 0.2                        # Maximum dropout threshold
#: The epoch interval value to gradually change dropout rate
DYNAMIC_DROP_OUT_REDUCTION_INTERVAL = 2                     # Interval for dropout rate reduction

#
# =============================================================================
# Dynamic augmentation parameters
# =============================================================================
#

#: The rate at which the probability of data augmentation is increased.
DYNAMIC_AUG_INC_RATE = 1.5                                  # Data augmentation increase rate
#: The maximum limit to which the probability of data augmentation can be increased.
DYNAMIC_AUG_MAX_THRESHOLD = 0.35                            # Maximum data augmentation threshold
#: The number of epochs to wait before increasing the probability of data augmentation.
DYNAMIC_AUG_INC_INTERVAL = 5                                # Interval for augmentation rate increase

#
# =============================================================================
# Model configuration
# =============================================================================
#

#: Deep learning framework to use for training and inference. Can be either 'pytorch' or 'tensorflow'.
DL_FRAMEWORK='pytorch'                                      # Deep learning framework to use for training and inference.
#: Name of the model to be used for training.
MODELNAME='YetAnotherTransformerClassifier'                 # Name of the model to be used for training.


#
# =============================================================================
# Data Files
# =============================================================================
#

#: CSV file name that contains the training dataset.
TRAIN_CSV_FILE = 'train.csv'                                # CSV file that contains the training dataset
#: JSON file that maps sign to prediction index.
MAP_JSON_FILE = 'sign_to_prediction_index_map.json'         # JSON file that maps sign to prediction index
#: Directory where training landmark files are stored.
LANDMARK_FILES = 'train_landmark_files'                     # Directory where training landmark files are stored
#: File that marks the preprocessing stage.
MARKER_FILE = 'preprocessed_data.marker'                    # File that marks the preprocessing stage
#: File that marks the data cleaning stage.
CLEANED_FILE = 'cleansed_data.marker'                       # File that marks the data cleaning stage

#
# =============================================================================
# Data Features
# =============================================================================
#

#: Number of rows per frame in the data.
ROWS_PER_FRAME = 543                                        # Number of rows per frame in the data
#: Number of features related to the face in the data.
FACE_FEATURES = 468                                         # Number of features related to the face in the data
#: Number of features related to the hand in the data.
HAND_FEATURES = 21                                          # Number of features related to the hand in the data
#: Number of features related to the pose in the data.
POSE_FEATURES = 33                                          # Number of features related to the pose in the data

#: Coordinate columns from the data to use for training.
COLUMNS_TO_USE = ["x", "y"]                                 # Coordinate columns from the data to use for training

#
# =============================================================================
# Landmark selection parameters
# =============================================================================
#

#: Start index for face feature in the data.
FACE_FEATURE_START = 0                                      # Start index for face feature in the data
#: Start index for pose feature in the data.
POSE_FEATURE_START = 489                                    # Start index for pose feature in the data
#: Start index for left hand feature in the data.
LEFT_HAND_FEATURE_START = 468                               # Start index for left hand feature in the data
#: Start index for right hand feature in the data.
RIGHT_HAND_FEATURE_START = 522                              # Start index for right hand feature in the data

#: Landmarks for Lips
FACE_LANDMARKS = np.array([                                 # Landmarks for face
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])
     
#: Landmarks for face
USEFUL_FACE_LANDMARKS = np.array(FACE_LANDMARKS.tolist())   # Landmarks for face
#: Landmarks for pose
USEFUL_POSE_LANDMARKS = np.arange(500, 514)                 # Landmarks for pose
#: Landmarks for left hand
USEFUL_LEFT_HAND_LANDMARKS = np.arange(468, 489)            # Landmarks for left hand
#: Landmarks for right hand
USEFUL_RIGHT_HAND_LANDMARKS = np.arange(522, 543)           # Landmarks for right hand
#: Landmarks for both hands
USEFUL_HAND_LANDMARKS = np.concatenate((USEFUL_LEFT_HAND_LANDMARKS, USEFUL_RIGHT_HAND_LANDMARKS), axis=0)
#: All Landmarks 
USEFUL_ALL_LANDMARKS = np.concatenate((USEFUL_FACE_LANDMARKS, USEFUL_HAND_LANDMARKS, USEFUL_POSE_LANDMARKS))

#: Count of facial features used
USED_FACE_FEATURES = len(USEFUL_FACE_LANDMARKS)
#: Count of body/pose features used
USED_POSE_FEATURES = len(USEFUL_POSE_LANDMARKS)
#: Count of hands features used (single hand only)
USED_HAND_FEATURES = len(USEFUL_LEFT_HAND_LANDMARKS)

#: Number of classes
N_CLASSES = 250                                             # Number of classes
#: Total number of used landmarks
N_LANDMARKS = USED_FACE_FEATURES + 2 * USED_HAND_FEATURES + USED_POSE_FEATURES
#: Number of dimensions used in training
N_DIMS = len(COLUMNS_TO_USE)

# Indices of landmarks that are used from the data
#: Indices of face landmarks that are used from the data.
FACE_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_FACE_LANDMARKS)).squeeze()
#: Indices of pose landmarks that are used from the data.
POSE_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_POSE_LANDMARKS)).squeeze()
#: Indices of hand landmarks that are used from the data.
HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_HAND_LANDMARKS)).squeeze()
#: Indices of left hand landmarks that are used from the data.
LEFT_HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_LEFT_HAND_LANDMARKS)).squeeze()
#: Indices of right hand landmarks that are used from the data.
RIGHT_HAND_INDICES = np.argwhere(np.isin(USEFUL_ALL_LANDMARKS, USEFUL_RIGHT_HAND_LANDMARKS)).squeeze()

#
# =============================================================================
# Data processing configuration
# =============================================================================
#

#: Number of missing values to interpolate in the data.
INTEREMOLATE_MISSING = 3
#: Skip data if there are this many consecutive zeros.
SKIP_CONSECUTIVE_ZEROS = 4
#: Size of the input data for the model.
INPUT_SIZE = 32
#: Maximum number of sequences in the input data.
MAX_SEQUENCES = 32
#: Minimum number of sequences in the input data.
MIN_SEQUEENCES = INPUT_SIZE / 4

