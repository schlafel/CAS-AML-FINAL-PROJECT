<div class="document">

<div class="documentwrapper">

<div class="bodywrapper">

<div class="body" role="main">

<div id="welcome-to-american-sign-language-recognition-s-documentation"
class="section">

# [American Sign Language Recognition](#)

### Navigation

Contents:

-   [Data Augmentations](augmentations.py)
-   [Training Callbacks](callbacks.py)
-   [Project Configuration](config.py)
-   [Data Utilities](data/data_utils.py)
-   [HyperParameter Search](hparam_search.py)
-   [Camera Stream Predictions](predict_on_camera.py)
-   [ASL Dataset](data/dataset.py)
-   [Data Utilities](dl_utils.py)
-   [Model Training](trainer.py)
-   [Data Visualizations](visualizations.py)
-   [Pytorch Models](pytorch_models.py)
-   [Tensorflow Models](tensorflow_models.py)
-   [Torch Lightning Models](lightning_models.py)

<div class="relations">

# Welcome to American Sign Language Recognition’s documentation![¶](#welcome-to-american-sign-language-recognition-s-documentation "Permalink to this headline")

This project aims to classify isolated American Sign Language (ASL) signs using deep learning techniques implemented in PyTorch. The dataset used for this project is provided by Google's "Isolated Sign Language Recognition" competition on Kaggle.

# Modules:
<div class="toctree-wrapper compound">

<div id="module-augmentations" class="section">

## Data Augmentations[¶](#module-augmentations "Permalink to this headline")

augmentations.frame\_dropout(*frames*, *dropout\_rate=0.05*)[¶](#augmentations.frame_dropout "Permalink to this definition")  
Randomly drop frames from the input landmark data.

Args:  
frames (numpy.ndarray): An array of landmarks data. dropout\_rate
(float): The proportion of frames to drop (default: 0.05).

Returns:  
numpy.ndarray: An array of landmarks with dropped frames.

&nbsp;

augmentations.mirror\_landmarks(*frames*)[¶](#augmentations.mirror_landmarks "Permalink to this definition")  
Invert/mirror landmark coordinates along the x-axis.

Args:  
frames (numpy.ndarray): An array of landmarks data.

Returns:  
numpy.ndarray: An array of inverted landmarks.

&nbsp;

augmentations.normalize(*frames*, *mn*, *std*)[¶](#augmentations.normalize "Permalink to this definition")  
Normalize the frames with a given mean and standard deviation.

Args:  
frames (numpy.ndarray): An array of landmarks data. mn (float): The mean
value for normalization. std (float): The standard deviation for
normalization.

Returns:  
numpy.ndarray: An array of normalized landmarks.

&nbsp;

augmentations.random\_rotation(*frames*, *max\_angle=10*)[¶](#augmentations.random_rotation "Permalink to this definition")  
Apply random rotation to landmark coordinates. (on X and Y only)

Args:  
frames (numpy.ndarray): An array of landmarks data. max\_angle (int):
The maximum rotation angle in degrees (default: 10).

Returns:  
numpy.ndarray: An array of landmarks with randomly rotated coordinates.

&nbsp;

augmentations.random\_scaling(*frames*, *scale\_range=(0.9, 1.1)*)[¶](#augmentations.random_scaling "Permalink to this definition")  
Apply random scaling to landmark coordinates.

Args:  
frames (numpy.ndarray): An array of landmarks data. scale\_range
(tuple): A tuple containing the minimum and maximum scaling factors
(default: (0.9, 1.1)).

Returns:  
numpy.ndarray: An array of landmarks with randomly scaled coordinates.

&nbsp;

augmentations.shift\_landmarks(*frames*, *max\_shift=0.01*)[¶](#augmentations.shift_landmarks "Permalink to this definition")  
Shift landmark coordinates randomly by a small amount.

Args:  
frames (numpy.ndarray): An array of landmarks data. max\_shift (float):
Maximum shift for the random shift (default: 0.01).

Returns:  
numpy.ndarray: An array of augmented landmarks.

&nbsp;

augmentations.standardize(*frames*)[¶](#augmentations.standardize "Permalink to this definition")  
Standardize the frames so that they have mean 0 and standard
deviation 1.

Args:  
frames (numpy.ndarray): An array of landmarks data.

Returns:  
numpy.ndarray: An array of standardized landmarks.

</div>

<div id="module-callbacks" class="section">

## Training Callbacks[¶](#module-callbacks "Permalink to this headline")

<div id="callbacks-description" class="section">

### Callbacks description[¶](#callbacks-description "Permalink to this headline")

This module contains callback codes which may be executed during
training. These callbacks are used to dynamically adjust the dropout
rate and data augmentation probability during the training process,
which can be useful techniques to prevent overfitting and increase the
diversity of the training data, potentially improving the model’s
performance.

The dropout\_callback function is designed to increase the dropout rate
of the model during the training process after a certain number of
epochs. The dropout rate is a regularization technique used to prevent
overfitting during the training process. The rate of dropout is
increased every few epochs based on a specified rate until it reaches a
specified maximum limit.

The augmentation\_increase\_callback: function is designed to increase
the probability of data augmentation applied to the dataset during the
training process after a certain number of epochs. Data augmentation is
a technique that can generate new training samples by applying
transformations to the existing data. The probability of data
augmentation is increased every few epochs based on a specified rate
until it reaches a specified maximum limit.

callbacks.augmentation\_increase\_callback(*trainer*, *aug\_increase\_rate=1.5*, *max\_limit=0.35*)[¶](#callbacks.augmentation_increase_callback "Permalink to this definition")  
A callback function designed to increase the probability of data
augmentation applied on the dataset during the training process. Data
augmentation is a technique that can generate new training samples by
applying transformations to the existing data.

The increase in data augmentation is performed every few epochs based on
‘DYNAMIC\_AUG\_INC\_INTERVAL’ until it reaches a specified maximum
limit.

Args:  
trainer: The object that contains the model and handles the training
process. aug\_increase\_rate: The rate at which data augmentation
probability is increased. Default is value of ‘DYNAMIC\_AUG\_INC\_RATE’
from config. max\_limit: The maximum limit to which data augmentation
probability can be increased. Default is value of
‘DYNAMIC\_AUG\_MAX\_THRESHOLD’ from config.

Returns:  
None

Functionality:  
Increases the probability of data augmentation applied on the dataset
after certain number of epochs defined by ‘DYNAMIC\_AUG\_INC\_INTERVAL’.

Parameters  
-   **trainer** – Trainer object handling the training process

-   **aug\_increase\_rate** – Rate at which to increase the data
    augmentation probability

-   **max\_limit** – Maximum allowable data augmentation probability

Returns  
None

Return type  
None

&nbsp;

callbacks.dropout\_callback(*trainer*, *dropout\_rate=1.1*, *max\_dropout=0.2*)[¶](#callbacks.dropout_callback "Permalink to this definition")  
A callback function designed to increase the dropout rate of the model
in training after a certain number of epochs. The dropout rate is a
regularization technique which helps in preventing overfitting during
the training process.

The rate of dropout is increased every few epochs based on the config
parameter (in config.py) ‘DYNAMIC\_DROP\_OUT\_REDUCTION\_INTERVAL’ until
a maximum threshold defined by ‘max\_dropout’. This function is usually
called after each epoch in the training process.

Args:  
trainer: The object that contains the model and handles the training
process. dropout\_rate: The rate at which the dropout rate is increased.
Default is value of ‘DYNAMIC\_DROP\_OUT\_REDUCTION\_RATE’ from config.
max\_dropout: The maximum limit to which dropout can be increased.
Default is value of ‘DYNAMIC\_DROP\_OUT\_MAX\_THRESHOLD’ from config.

Returns:  
None

Functionality:  
Increases the dropout rate of all nn.Dropout modules in the model after
certain number of epochs defined by
‘DYNAMIC\_DROP\_OUT\_REDUCTION\_INTERVAL’.

Parameters  
-   **trainer** – Trainer object handling the training process

-   **dropout\_rate** – Rate at which to increase the dropout rate

-   **max\_dropout** – Maximum allowable dropout rate

Returns  
None

Return type  
None

</div>

</div>

<div id="module-config" class="section">

## Project Configuration[¶](#module-config "Permalink to this headline")

<div id="project-configuration-description" class="section">

### Project configuration description[¶](#project-configuration-description "Permalink to this headline")

This configuration is created allows for easy tuning of your machine
learning model’s parameters and setup. The device on which the model
runs, the paths for various resources, the seed for random number
generation, hyperparameters for model training, and much more and
quickly be change and configured. This makes your setup flexible and
easy to adapt for various experiments and environments

config.BATCH\_SIZE* = 128*[¶](#config.BATCH_SIZE "Permalink to this definition")  
Training Batch Size

&nbsp;

config.CHECKPOINT\_DIR* = 'checkpoints/'*[¶](#config.CHECKPOINT_DIR "Permalink to this definition")  
Checkpoint files Directory path

&nbsp;

config.CLEANED\_FILE* = 'cleansed\_data.marker'*[¶](#config.CLEANED_FILE "Permalink to this definition")  
File that marks the data cleaning stage.

&nbsp;

config.COLUMNS\_TO\_USE* = \['x', 'y'\]*[¶](#config.COLUMNS_TO_USE "Permalink to this definition")  
Coordinate columns from the data to use for training.

&nbsp;

config.DATA\_DIR* = 'data/'*[¶](#config.DATA_DIR "Permalink to this definition")  
Data files Directory path

&nbsp;

config.DEVICE* = 'cpu'*[¶](#config.DEVICE "Permalink to this definition")  
Setting the device for training, ‘cuda’ if a CUDA-compatible GPU is
available, ‘mps’ if multiple processors are available, ‘cpu’ if none of
the above.

&nbsp;

config.DL\_FRAMEWORK* = 'pytorch'*[¶](#config.DL_FRAMEWORK "Permalink to this definition")  
Deep learning framework to use for training and inference. Can be either
‘pytorch’ or ‘tensorflow’.

&nbsp;

config.DYNAMIC\_AUG\_INC\_INTERVAL* = 5*[¶](#config.DYNAMIC_AUG_INC_INTERVAL "Permalink to this definition")  
The number of epochs to wait before increasing the probability of data
augmentation.

&nbsp;

config.DYNAMIC\_AUG\_INC\_RATE* = 1.5*[¶](#config.DYNAMIC_AUG_INC_RATE "Permalink to this definition")  
The rate at which the probability of data augmentation is increased.

&nbsp;

config.DYNAMIC\_AUG\_MAX\_THRESHOLD* = 0.35*[¶](#config.DYNAMIC_AUG_MAX_THRESHOLD "Permalink to this definition")  
The maximum limit to which the probability of data augmentation can be
increased.

&nbsp;

config.DYNAMIC\_DROP\_OUT\_INIT\_RATE* = 0.01*[¶](#config.DYNAMIC_DROP_OUT_INIT_RATE "Permalink to this definition")  
The value of initial low dropouts rate

&nbsp;

config.DYNAMIC\_DROP\_OUT\_MAX\_THRESHOLD* = 0.2*[¶](#config.DYNAMIC_DROP_OUT_MAX_THRESHOLD "Permalink to this definition")  
The max value of dynamic dropouts

&nbsp;

config.DYNAMIC\_DROP\_OUT\_REDUCTION\_INTERVAL* = 2*[¶](#config.DYNAMIC_DROP_OUT_REDUCTION_INTERVAL "Permalink to this definition")  
The epoch interval value to gradually change dropout rate

&nbsp;

config.DYNAMIC\_DROP\_OUT\_REDUCTION\_RATE* = 1.1*[¶](#config.DYNAMIC_DROP_OUT_REDUCTION_RATE "Permalink to this definition")  
The value to increase dropouts by

&nbsp;

config.EARLY\_STOP\_METRIC* = 'accuracy'*[¶](#config.EARLY_STOP_METRIC "Permalink to this definition")  
Which metric should be used for early stopping loss/accuracy

&nbsp;

config.EARLY\_STOP\_MODE* = 'max'*[¶](#config.EARLY_STOP_MODE "Permalink to this definition")  
What is the mode? min/max

&nbsp;

config.EARLY\_STOP\_PATIENCE* = 10*[¶](#config.EARLY_STOP_PATIENCE "Permalink to this definition")  
The number of epochs to wait for improvement in the validation loss
before stopping training

&nbsp;

config.EARLY\_STOP\_TOLERENCE* = 0.001*[¶](#config.EARLY_STOP_TOLERENCE "Permalink to this definition")  
The value of loss as margin to tolerate

&nbsp;

config.EPOCHS* = 50*[¶](#config.EPOCHS "Permalink to this definition")  
Training Number of epochs

&nbsp;

config.FACE\_FEATURES* = 468*[¶](#config.FACE_FEATURES "Permalink to this definition")  
Number of features related to the face in the data.

&nbsp;

config.FACE\_FEATURE\_START* = 0*[¶](#config.FACE_FEATURE_START "Permalink to this definition")  
Start index for face feature in the data.

&nbsp;

config.FACE\_INDICES* = array(\[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,        34, 35, 36, 37, 38, 39\], dtype=int64)*[¶](#config.FACE_INDICES "Permalink to this definition")  
Indices of face landmarks that are used from the data.

&nbsp;

config.FACE\_LANDMARKS* = array(\[ 61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291, 146,  91,        181,  84,  17, 314, 405, 321, 375,  78, 191,  80,  81,  82,  13,        312, 311, 310, 415,  95,  88, 178,  87,  14, 317, 402, 318, 324,        308\])*[¶](#config.FACE_LANDMARKS "Permalink to this definition")  
Landmarks for Lips

&nbsp;

config.HAND\_FEATURES* = 21*[¶](#config.HAND_FEATURES "Permalink to this definition")  
Number of features related to the hand in the data.

&nbsp;

config.HAND\_INDICES* = array(\[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,        74, 75, 76, 77, 78, 79, 80, 81\], dtype=int64)*[¶](#config.HAND_INDICES "Permalink to this definition")  
Indices of hand landmarks that are used from the data.

&nbsp;

config.INPUT\_SIZE* = 32*[¶](#config.INPUT_SIZE "Permalink to this definition")  
Size of the input data for the model.

&nbsp;

config.INTEREMOLATE\_MISSING* = 3*[¶](#config.INTEREMOLATE_MISSING "Permalink to this definition")  
Number of missing values to interpolate in the data.

&nbsp;

config.LANDMARK\_FILES* = 'train\_landmark\_files'*[¶](#config.LANDMARK_FILES "Permalink to this definition")  
Directory where training landmark files are stored.

&nbsp;

config.LEARNING\_RATE* = 0.001*[¶](#config.LEARNING_RATE "Permalink to this definition")  
Training Learning rate

&nbsp;

config.LEFT\_HAND\_FEATURE\_START* = 468*[¶](#config.LEFT_HAND_FEATURE_START "Permalink to this definition")  
Start index for left hand feature in the data.

&nbsp;

config.LEFT\_HAND\_INDICES* = array(\[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,        57, 58, 59, 60\], dtype=int64)*[¶](#config.LEFT_HAND_INDICES "Permalink to this definition")  
Indices of left hand landmarks that are used from the data.

&nbsp;

config.MAP\_JSON\_FILE* = 'sign\_to\_prediction\_index\_map.json'*[¶](#config.MAP_JSON_FILE "Permalink to this definition")  
JSON file that maps sign to prediction index.

&nbsp;

config.MARKER\_FILE* = 'preprocessed\_data.marker'*[¶](#config.MARKER_FILE "Permalink to this definition")  
File that marks the preprocessing stage.

&nbsp;

config.MAX\_SEQUENCES* = 32*[¶](#config.MAX_SEQUENCES "Permalink to this definition")  
Maximum number of sequences in the input data.

&nbsp;

config.MIN\_SEQUEENCES* = 8.0*[¶](#config.MIN_SEQUEENCES "Permalink to this definition")  
Minimum number of sequences in the input data.

&nbsp;

config.MODELNAME* = 'YetAnotherTransformerClassifier'*[¶](#config.MODELNAME "Permalink to this definition")  
Name of the model to be used for training.

&nbsp;

config.MODEL\_DIR* = 'models/'*[¶](#config.MODEL_DIR "Permalink to this definition")  
Model files Directory path

&nbsp;

config.N\_CLASSES* = 250*[¶](#config.N_CLASSES "Permalink to this definition")  
Number of classes

&nbsp;

config.N\_DIMS* = 2*[¶](#config.N_DIMS "Permalink to this definition")  
Number of dimensions used in training

&nbsp;

config.N\_LANDMARKS* = 96*[¶](#config.N_LANDMARKS "Permalink to this definition")  
Total number of used landmarks

&nbsp;

config.OUT\_DIR* = 'out/'*[¶](#config.OUT_DIR "Permalink to this definition")  
Output files Directory path

&nbsp;

config.POSE\_FEATURES* = 33*[¶](#config.POSE_FEATURES "Permalink to this definition")  
Number of features related to the pose in the data.

&nbsp;

config.POSE\_FEATURE\_START* = 489*[¶](#config.POSE_FEATURE_START "Permalink to this definition")  
Start index for pose feature in the data.

&nbsp;

config.POSE\_INDICES* = array(\[82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95\],       dtype=int64)*[¶](#config.POSE_INDICES "Permalink to this definition")  
Indices of pose landmarks that are used from the data.

&nbsp;

config.PROCESSED\_DATA\_DIR* = 'data/processed/'*[¶](#config.PROCESSED_DATA_DIR "Permalink to this definition")  
Processed Data files Directory path

&nbsp;

config.RAW\_DATA\_DIR* = 'data/raw/'*[¶](#config.RAW_DATA_DIR "Permalink to this definition")  
Raw Data files Directory path

&nbsp;

config.RIGHT\_HAND\_FEATURE\_START* = 522*[¶](#config.RIGHT_HAND_FEATURE_START "Permalink to this definition")  
Start index for right hand feature in the data.

&nbsp;

config.RIGHT\_HAND\_INDICES* = array(\[61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,        78, 79, 80, 81\], dtype=int64)*[¶](#config.RIGHT_HAND_INDICES "Permalink to this definition")  
Indices of right hand landmarks that are used from the data.

&nbsp;

config.ROOT\_PATH* = 'C:\\\\Users\\\\tgdimas1\\\\git\\\\CAS-AML-FINAL-PROJECT\\\\src\\\\../'*[¶](#config.ROOT_PATH "Permalink to this definition")  
Root directory

&nbsp;

config.ROWS\_PER\_FRAME* = 543*[¶](#config.ROWS_PER_FRAME "Permalink to this definition")  
Number of rows per frame in the data.

&nbsp;

config.RUNS\_DIR* = 'runs/'*[¶](#config.RUNS_DIR "Permalink to this definition")  
Run files Directory path

&nbsp;

config.SEED* = 0*[¶](#config.SEED "Permalink to this definition")  
Set Random Seed

&nbsp;

config.SKIP\_CONSECUTIVE\_ZEROS* = 4*[¶](#config.SKIP_CONSECUTIVE_ZEROS "Permalink to this definition")  
Skip data if there are this many consecutive zeros.

&nbsp;

config.SRC\_DIR* = 'src/'*[¶](#config.SRC_DIR "Permalink to this definition")  
Source files Directory path

&nbsp;

config.TEST\_SIZE* = 0.05*[¶](#config.TEST_SIZE "Permalink to this definition")  
Testing Test set size

&nbsp;

config.TRAIN\_CSV\_FILE* = 'train.csv'*[¶](#config.TRAIN_CSV_FILE "Permalink to this definition")  
CSV file name that contains the training dataset.

&nbsp;

config.TRAIN\_SIZE* = 0.9*[¶](#config.TRAIN_SIZE "Permalink to this definition")  
Training Train set split size

&nbsp;

config.TUNE\_HP* = True*[¶](#config.TUNE_HP "Permalink to this definition")  
Tune hyperparameters

&nbsp;

config.USED\_FACE\_FEATURES* = 40*[¶](#config.USED_FACE_FEATURES "Permalink to this definition")  
Count of facial features used

&nbsp;

config.USED\_HAND\_FEATURES* = 21*[¶](#config.USED_HAND_FEATURES "Permalink to this definition")  
Count of hands features used (single hand only)

&nbsp;

config.USED\_POSE\_FEATURES* = 14*[¶](#config.USED_POSE_FEATURES "Permalink to this definition")  
Count of body/pose features used

&nbsp;

config.USEFUL\_ALL\_LANDMARKS* = array(\[ 61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291, 146,  91,        181,  84,  17, 314, 405, 321, 375,  78, 191,  80,  81,  82,  13,        312, 311, 310, 415,  95,  88, 178,  87,  14, 317, 402, 318, 324,        308, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,        480, 481, 482, 483, 484, 485, 486, 487, 488, 522, 523, 524, 525,        526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538,        539, 540, 541, 542, 500, 501, 502, 503, 504, 505, 506, 507, 508,        509, 510, 511, 512, 513\])*[¶](#config.USEFUL_ALL_LANDMARKS "Permalink to this definition")  
All Landmarks

&nbsp;

config.USEFUL\_FACE\_LANDMARKS* = array(\[ 61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291, 146,  91,        181,  84,  17, 314, 405, 321, 375,  78, 191,  80,  81,  82,  13,        312, 311, 310, 415,  95,  88, 178,  87,  14, 317, 402, 318, 324,        308\])*[¶](#config.USEFUL_FACE_LANDMARKS "Permalink to this definition")  
Landmarks for face

&nbsp;

config.USEFUL\_HAND\_LANDMARKS* = array(\[468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,        481, 482, 483, 484, 485, 486, 487, 488, 522, 523, 524, 525, 526,        527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,        540, 541, 542\])*[¶](#config.USEFUL_HAND_LANDMARKS "Permalink to this definition")  
Landmarks for both hands

&nbsp;

config.USEFUL\_LEFT\_HAND\_LANDMARKS* = array(\[468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,        481, 482, 483, 484, 485, 486, 487, 488\])*[¶](#config.USEFUL_LEFT_HAND_LANDMARKS "Permalink to this definition")  
Landmarks for left hand

&nbsp;

config.USEFUL\_POSE\_LANDMARKS* = array(\[500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512,        513\])*[¶](#config.USEFUL_POSE_LANDMARKS "Permalink to this definition")  
Landmarks for pose

&nbsp;

config.USEFUL\_RIGHT\_HAND\_LANDMARKS* = array(\[522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534,        535, 536, 537, 538, 539, 540, 541, 542\])*[¶](#config.USEFUL_RIGHT_HAND_LANDMARKS "Permalink to this definition")  
Landmarks for right hand

&nbsp;

config.VALID\_SIZE* = 0.05*[¶](#config.VALID_SIZE "Permalink to this definition")  
Training Validation set size

</div>

</div>

<div id="module-data.data_utils" class="section">

## Data Utilities[¶](#module-data.data_utils "Permalink to this headline")

<div id="data-processing-utils-description" class="section">

### Data processing Utils description[¶](#data-processing-utils-description "Permalink to this headline")

This module handles the loading and preprocessing of data. It is
specifically tailored for loading ASL sign language dataset where the
raw data includes information about the position of hands, face, and
body over time.

ASL stands for American Sign Language, which is a natural language used
by individuals who are deaf or hard of hearing to communicate through
hand gestures and facial expressions.

The dataset consists of sequences of frames, where each frame contains
multiple “landmarks”. Each of these landmarks has multiple features,
such as coordinates. The landmarks may represent various aspects of
human body, such as facial features, hand positions, and body pose.

This module is used to process the raw data, to create a uniform dataset
where all sequences are of the same length and all missing values have
been handled in a way that maintains the integrity of the data. This
involves steps like detecting and removing empty frames, selecting
specific landmarks, resizing sequences and handling NaN values.

data.data\_utils.calculate\_avg\_landmark\_positions(*dataset*)[¶](#data.data_utils.calculate_avg_landmark_positions "Permalink to this definition")  
Calculate the average landmark positions for left-hand, right-hand, and
face landmarks for each sign in the dataset. The purpose of this
function is to compute the average positions of landmarks for left-hand,
right-hand, and face for each sign in the training dataset.

Returns: List : Containing a dictionary with average x/y positions with
keys - ‘left\_hand’ - ‘right\_hand’ - ‘face’

Functionality: - The function takes an ASLDataset object as an input,
which contains the training data. - It calculates the average landmark
positions for left-hand, right-hand, and face landmarks for each sign in
the dataset. - The function returns a list containing a dictionary with
average x/y positions with keys ‘left\_hand’, ‘right\_hand’, and ‘face’
for each sign.

Parameters  
**dataset**
([*ASL\_DATASET*](index.html#data.dataset.ASL_DATASET "data.dataset.ASL_DATASET"))
– The ASL dataset object containing the training data.

Returns  
A list containing a dictionary with average x/y positions with keys
‘left\_hand’, ‘right\_hand’, and ‘face’

for each sign. :rtype: List\[Dict\[str, np.ndarray\]\]

&nbsp;

data.data\_utils.calculate\_landmark\_length\_stats()[¶](#data.data_utils.calculate_landmark_length_stats "Permalink to this definition")  
Calculate statistics of landmark lengths for each sign type.

Returns: dict: A dictionary of landmark lengths for each sign type
containing: - minimum - maximum - mean - median - standard deviation

Functionality: - The function reads the CSV file. - It groups the
DataFrame by sign. - An empty dictionary is created to store average
landmarks for each sign type. - The function loops through each unique
sign and its corresponding rows in the grouped DataFrame. - For each
sign, it initializes a list to store the length of landmarks for each
example of the current sign. - It loops through each row of the current
sign type, loads the data, and adds the length of landmarks of the
current example to the list of current sign data. - The function
calculates the minimum, maximum, mean, standard deviation, and median of
the landmarks for the current sign and updates the dictionary. - The
resulting dictionary containing average landmarks for each sign type is
returned.

Returns  
A dictionary of landmark lengths for each sign type containing minimum,
maximum, mean, median & standard

deviation :rtype: dict

&nbsp;

data.data\_utils.create\_data\_loaders(*asl\_dataset*, *train\_size=0.9*, *valid\_size=0.05*, *test\_size=0.05*, *batch\_size=128*, *random\_state=0*, *dl\_framework='pytorch'*, *num\_workers=8*)[¶](#data.data_utils.create_data_loaders "Permalink to this definition")  
Split the ASL dataset into training, validation, and testing sets and
create data loaders for each set.

Args: asl\_dataset (ASLDataset): The ASL dataset to load data from.
train\_size (float, optional): The proportion of the dataset to include
in the training set. Defaults to 0.8. valid\_size (float, optional): The
proportion of the dataset to include in the validation set. Defaults to
0.1. test\_size (float, optional): The proportion of the dataset to
include in the testing set. Defaults to 0.1. batch\_size (int,
optional): The number of samples per batch to load. Defaults to
BATCH\_SIZE. random\_state (int, optional): The seed used by the random
number generator for shuffling the data. Defaults to SEED.

Returns: tuple of DataLoader: A tuple containing the data loaders for
training, validation, and testing sets.

Parameters  
-   **asl\_dataset** (*ASLDataset*) – The ASL dataset to load data from.

-   **train\_size** (*float*) – The proportion of the dataset to include
    in the training set.

-   **valid\_size** (*float*) – The proportion of the dataset to include
    in the validation set.

-   **test\_size** (*float*) – The proportion of the dataset to include
    in the testing set.

-   **batch\_size** (*int*) – The number of samples per batch to load.

-   **random\_state** (*int*) – The seed used by the random number
    generator for shuffling the data.

Returns  
A tuple containing the data loaders for training, validation, and
testing sets.

Return type  
tuple of DataLoader

&nbsp;

data.data\_utils.interpolate\_missing\_values(*arr*, *max\_gap=3*)[¶](#data.data_utils.interpolate_missing_values "Permalink to this definition")  
This function provides a solution for handling missing values in the
data array. It interpolates these missing values, filling them with
plausible values that maintain the overall data integrity. The function
uses a linear interpolation method that assumes a straight line between
the two points on either side of the gap. The maximum gap size for which
interpolation should be performed is also configurable.

AThe function takes two arguments - an array with missing values, and a
maximum gap size for interpolation. If the size of the gap (i.e., number
of consecutive missing values) is less than or equal to this specified
maximum gap size, the function will fill it with interpolated values.
This ensures that the data maintains its continuity without making too
far-fetched estimations for larger gaps.

Args:  
arr (np.ndarray): Input array with missing values. max\_gap (int,
optional): Maximum gap to fill. Defaults to INTEREMOLATE\_MISSING.

Returns:  
np.ndarray: Array with missing values interpolated.

Functionality:  
Interpolates missing values in the array. The function fills gaps of up
to a maximum size with interpolated values, maintaining data integrity
and continuity.

Returns  
Array with missing values interpolated.

Return type  
np.ndarray

Parameters  
-   **arr** (*np.ndarray*) – Input array with missing values.

-   **max\_gap** (*int*) – Maximum gap to fill.

This function uses linear interpolation to fill the missing values.
Other forms of interpolation such as polynomial or spline may provide
better results for specific types of data. It is also worth noting that
no imputation method can fully recover original data, and as such,
results should be interpreted with caution when working with imputed
data.

&nbsp;

data.data\_utils.load\_relevant\_data\_subset(*pq\_path*)[¶](#data.data_utils.load_relevant_data_subset "Permalink to this definition")  
This function serves a key role in handling data in our pipeline by
loading only a subset of the relevant data from a given path. The
primary purpose of this is to reduce memory overhead when working with
large datasets. The implementation relies on efficient data loading
strategies, leveraging the speed of Parquet file format and the ability
to read in only necessary chunks of data instead of the whole dataset.

The function takes as input a string which represents the path to the
data file. It makes use of pandas’ parquet read function to read the
data file. This function is particularly suited for reading large
datasets as it allows for efficient on-disk storage and fast query
capabilities. The function uses PyArrow library as the engine for
reading the parquet files which ensures efficient and fast reading of
data. After reading the data, the function selects the relevant subset
based on certain criteria, which is task specific.

Args:  
pq\_path (str): Path to the data file.

Returns:  
np.ndarray: Subset of the relevant data as a NumPy array.

Functionality:  
Loads a subset of the relevant data from a given path.

Returns  
Subset of the relevant data.

Return type  
np.ndarray

Parameters  
**pq\_path** (*str*) – Path to the data file.

The function assumes that the data file is in parquet format and the
necessary libraries for reading parquet files are installed. It also
assumes that the path provided is a valid path to the data file.

&nbsp;

data.data\_utils.preprocess\_data(*landmarks*)[¶](#data.data_utils.preprocess_data "Permalink to this definition")  
This function preprocesses the input data by applying similar steps as
the preprocess\_data\_to\_same\_size function, but with the difference
that it does not interpolate missing values. The function again targets
to adjust the size of the input data to align with the INPUT\_SIZE. It
selects only non-empty frames and follows similar strategies of padding,
repeating, and pooling the data for size alignment.

Args:  
landmarks (np.ndarray): The input array with landmarks data.

Returns:  
Tuple\[np.ndarray, int\]: A tuple containing processed landmark data and
the final size of the data.

Parameters  
**landmarks** (*np.ndarray*) – The input array with landmarks data.

Returns  
A tuple containing processed landmark data and the final size of the
data.

Return type  
Tuple\[np.ndarray, int\]

&nbsp;

data.data\_utils.preprocess\_data\_item(*raw\_landmark\_path*, *targets\_sign*)[¶](#data.data_utils.preprocess_data_item "Permalink to this definition")  
The function preprocesses landmark data for a single file. The process
involves applying transformations to raw landmark data to convert it
into a form more suitable for machine learning models. The
transformations may include normalization, scaling, etc. The target sign
associated with the landmark data is also taken as input.

This function is a handy function to process all landmark aequences on a
particular location. This will come in handy while testing where
individual sequences may be provided

Args:  
raw\_landmark\_path: Path to the raw landmark file targets\_sign: The
target sign for the given landmark data

Returns: dict: A dictionary containing the preprocessed landmarks,
target, and size.

Functionality: - The function reads the parquet file and processes the
data. - It filters columns to include only frame, type, landmark\_index,
x, and y. - The function then filters face mesh landmarks and pose
landmarks based on the predefined useful landmarks. - Landmarks data is
pivoted to have a multi-level column structure on landmark type and
frame sequence ids. - Missing values are interpolated using linear
interpolation, and any remaining missing values are filled with 0. - The
function rearranges columns and calculates the number of frames in the
data. - X and Y coordinates are brought together, and a dictionary with
the processed data is created and returned.

Parameters  
-   **raw\_landmark\_path** (*str*) – Path to the raw landmark file.

-   **targets\_sign** (*int*) – The target sign for the given landmark
    data.

Returns  
A dictionary containing the preprocessed landmarks, target, and size.

Return type  
dict

&nbsp;

data.data\_utils.preprocess\_data\_to\_same\_size(*landmarks*)[¶](#data.data_utils.preprocess_data_to_same_size "Permalink to this definition")  
This function preprocesses the input data to ensure all data arrays have
the same size, specified by the global INPUT\_SIZE variable. This
uniform size is necessary for subsequent processing and analysis stages,
particularly those involving machine learning models which often require
consistent input sizes. The preprocessing involves several steps,
including handling missing values, upsampling, and reshaping arrays. It
begins by interpolating any missing values, and then it subsets the data
by selecting only non-empty frames. Various strategies are applied to
align the data size to the desired INPUT\_SIZE, including padding,
repeating, and pooling the data.

Args:  
landmarks (np.ndarray): The input array with landmarks data.

Returns:  
Tuple\[np.ndarray, int, int, int\]: A tuple containing processed
landmark data, the set input size, the number of original frames, and
the number of frames after preprocessing.

Parameters  
**landmarks** (*np.ndarray*) – The input array with landmarks data.

Returns  
A tuple containing processed landmark data, the set input size, the
number of original frames, and the

number of frames after preprocessing. :rtype: Tuple\[np.ndarray, int,
int, int\]

&nbsp;

data.data\_utils.preprocess\_raw\_data(*sample=100000*)[¶](#data.data_utils.preprocess_raw_data "Permalink to this definition")  
Preprocesses the raw data, saves it as numpy arrays into processed data
directory and updates the metadata CSV file.

This method preprocess\_data preprocesses the data for easier and faster
loading during training time. The data is processed and stored in
PROCESSED\_DATA\_DIR if not already done.

This function is responsible for preprocessing raw data. The primary
functionality involves converting raw data into a format more suitable
for the machine learning pipeline, namely NumPy arrays. The function
operates on a sample of data, allowing for efficient processing of large
datasets in manageable chunks. Additionally, this function also takes
care of persisting the preprocessed data for future use and updates the
metadata accordingly.

Args: sample (int): Number of samples to preprocess. Default is 100000.

Functionality: - The function reads the metadata CSV file for training
data to obtain a dictionary that maps target values to integer
indices. - It then reads the training data CSV file and generates the
absolute path to locate landmark files. - Next, it keeps text signs and
their respective indices and initializes a list to store the processed
data. - The data is then processed and stored in the list by iterating
over each file path in the training data and reading in the parquet file
for that file path. - The landmark data is then processed and padded to
have a length of max\_seq\_length. - Finally, a dictionary with the
processed data is created and added to the list. - The processed data is
saved to disk using the np.save method and the saved file is printed.

Parameters  
**sample** (*int,* *optional,* *default: 100000*) – Number of samples to
preprocess.

Returns  
None

<div class="admonition note">

Note

If the preprocessed data already exists, the function prints
“Preprocessed data found. Skipping…” and exits.

</div>

&nbsp;

data.data\_utils.remove\_outlier\_or\_missing\_data(*landmark\_len\_dict*)[¶](#data.data_utils.remove_outlier_or_missing_data "Permalink to this definition")  
This function removes rows from the training data that contain missing
or outlier landmark data. It takes as input a dictionary containing the
statistics of landmark lengths for each sign type. The function
processes the training data and removes rows with missing or outlier
landmark data. The function also includes a nested function
‘has\_consecutive\_zeros’ which checks for consecutive frames where X
and Y coordinates are both zero. If a cleansing marker file exists, it
skips the process, indicating that the data is already cleaned.

Functionality:  
This function takes a dictionary with the statistics of landmark lengths
per sign type and uses it to identify outlier sequences. It removes any
rows with missing or outlier landmark data. An outlier sequence is
defined as one that is either less than a third of the median length or
more than two standard deviations away from the mean length. A row is
also marked for deletion if the corresponding landmark file is missing
or if the sign’s left-hand or right-hand landmarks contain more than a
specified number of consecutive zeros.

Args:  
landmark\_len\_dict (dict): A dictionary containing the statistics of
landmark lengths for each sign type.

Returns:  
None

Parameters  
**landmark\_len\_dict** (*dict*) – A dictionary containing the
statistics of landmark lengths for each sign type.

Returns  
None, the function doesn’t return anything. It modifies data in-place.

&nbsp;

data.data\_utils.remove\_unusable\_data()[¶](#data.data_utils.remove_unusable_data "Permalink to this definition")  
This function checks the existing training data for unusable instances,
like missing files or data that is smaller than the set minimum sequence
length. If unusable data is found, it is removed from the system, both
in terms of files and entries in the training dataframe. The dataframe
is updated and saved back to the disk. If a cleansing marker file
exists, it skips the process, indicating that the data is already
cleaned.

Functionality:  
The function iterates through the DataFrame rows, attempting to load and
check each landmark file specified in the row’s path. If the file is
missing or if the file’s usable size is less than a predefined
threshold, the function deletes the corresponding landmark file and
marks the row for deletion in the DataFrame. At the end, the function
removes all marked rows from the DataFrame, updates it and saves it to
the disk.

Returns:  
None

Returns  
None, the function doesn’t return anything. It modifies data in-place.

</div>

</div>

<div id="module-hparam_search" class="section">

## HyperParameter Search[¶](#module-hparam_search "Permalink to this headline")

Examples using MLfowLoggerCallback and setup\_mlflow.

*class *hparam\_search.Trainer\_HparamSearch(*modelname='YetAnotherTransformerClassifier'*, *dataset=&lt;class 'data.dataset.ASL\_DATASET'&gt;*, *patience=10*)[¶](#hparam_search.Trainer_HparamSearch "Permalink to this definition")  
\_\_init\_\_(*modelname='YetAnotherTransformerClassifier'*, *dataset=&lt;class 'data.dataset.ASL\_DATASET'&gt;*, *patience=10*)[¶](#hparam_search.Trainer_HparamSearch.__init__ "Permalink to this definition")  
Initializes the Trainer class with the specified parameters.

This method initializes various components needed for the training
process. This includes the model specified by the model name, the
dataset with optional data augmentation and dropout, data loaders for
the training, validation, and test sets, a SummaryWriter for logging,
and a path for saving model checkpoints.

1.  The method first retrieves the specified model and its parameters.

2.  It then initializes the dataset and the data loaders.

3.  It sets up metrics for early stopping and a writer for logging.

4.  Finally, it prepares a directory for saving model checkpoints.

Args:  
modelname (str): The name of the model to be used for training. dataset
(Dataset): The dataset to be used. patience (int): The number of epochs
with no improvement after which training will be stopped.
enableAugmentationDropout (bool): If True, enable data augmentation
dropout. augmentation\_threshold (float): The threshold for data
augmentation.

Functionality:  
This method initializes various components, such as the model, dataset,
data loaders, logging writer, and checkpoint path, required for the
training process.

Parameters  
-   **modelname** (*str*) – The name of the model for training.

-   **dataset** (*Dataset*) – The dataset for training.

-   **patience** (*int*) – The number of epochs with no improvement
    after which training will be stopped.

-   **enableAugmentationDropout** (*bool*) – If True, enable data
    augmentation dropout.

-   **augmentation\_threshold** (*float*) – The threshold for data
    augmentation.

Return type  
None

<div class="admonition note">

Note

This method only initializes the Trainer class. The actual training is
done by calling the train() method.

</div>

<div class="admonition warning">

Warning

Make sure the specified model name corresponds to an actual model in
your project’s models directory.

</div>

train(*n\_epochs=50*)[¶](#hparam_search.Trainer_HparamSearch.train "Permalink to this definition")  
Trains the model for a specified number of epochs.

This method manages the main training loop of the model. For each epoch,
it performs several steps. It first puts the model into training mode
and loops over the training dataset, calculating the loss and accuracy
for each batch and optimizing the model parameters. It logs these
metrics and updates a progress bar. At the end of each epoch, it
evaluates the model on the validation set and checks whether early
stopping criteria have been met. If the early stopping metric has
improved, it saves the current model and its parameters. If not, it
increments a counter and potentially stops training if the counter
exceeds the allowed patience. Finally, it steps the learning rate
scheduler and calls any registered callbacks.

1.  The method first puts the model into training mode and initializes
    some lists and counters.

2.  Then it enters the main loop over the training data, updating the
    model and logging metrics.

3.  It evaluates the model on the validation set and checks the early
    stopping criteria.

4.  If the criteria are met, it saves the model and its parameters; if
    not, it increments a patience counter.

5.  It steps the learning rate scheduler and calls any callbacks.

Args:  
n\_epochs (int): The number of epochs for which the model should be
trained.

Functionality:  
This method coordinates the training of the model over a series of
epochs, handling batch-wise loss computation, backpropagation,
optimization, validation, early stopping, and model checkpoint saving.

Parameters  
**n\_epochs** (*int*) – Number of epochs for training.

Returns  
None

Return type  
None

<div class="admonition note">

Note

This method modifies the state of the model and its optimizer, as well
as various attributes of the Trainer instance itself.

</div>

<div class="admonition warning">

Warning

If you set the patience value too low in the constructor, the model
might stop training prematurely.

</div>

</div>

<div id="module-predict_on_camera" class="section">

## Camera Stream Predictions[¶](#module-predict_on_camera "Permalink to this headline")

predict\_on\_camera.convert\_mp\_to\_df(*results*)[¶](#predict_on_camera.convert_mp_to_df "Permalink to this definition")  

&nbsp;

predict\_on\_camera.show\_camera\_feed(*model*, *LAST\_FRAMES=32*)[¶](#predict_on_camera.show_camera_feed "Permalink to this definition")  

</div>

<div id="module-data.dataset" class="section">

## ASL Dataset[¶](#module-data.dataset "Permalink to this headline")

<div id="asl-dataset-description" class="section">

### ASL Dataset description[¶](#asl-dataset-description "Permalink to this headline")

This file contains the ASL\_DATASET class which serves as the dataset
module for American Sign Language (ASL) data. The ASL\_DATASET is
designed to load, preprocess, augment, and serve the dataset for model
training and validation. This class provides functionalities such as
loading the dataset from disk, applying transformations, data
augmentation techniques, and an interface to access individual data
samples.

<div class="admonition note">

Note

This dataset class expects data in a specific format. Detailed
explanations and expectations about input data are provided in
respective method docstrings.

</div>

*class *data.dataset.ASL\_DATASET(*metadata\_df=None*, *transform=None*, *max\_seq\_length=32*, *augment=False*, *augmentation\_threshold=0.1*, *enableDropout=True*)[¶](#data.dataset.ASL_DATASET "Permalink to this definition")  
A dataset class for the ASL dataset.

The ASL\_DATASET class represents a dataset of American Sign Language
(ASL) gestures, where each gesture corresponds to a word or phrase. This
class provides functionalities to load the dataset, apply
transformations, augment the data, and yield individual data samples for
model training and validation.

\_\_getitem\_\_(*idx*)[¶](#data.dataset.ASL_DATASET.__getitem__ "Permalink to this definition")  
Get an item from the dataset by index.

This method returns a data sample from the dataset based on a provided
index. It handles reading of the processed data file, applies
transformations and augmentations (if set), and pads the data to match
the maximum sequence length. It returns the preprocessed landmarks and
corresponding target as a tuple.

Args:  
idx (int): The index of the item to retrieve.

Returns:  
tuple: A tuple containing the landmarks and target for the item.

Functionality:  
Get a single item from the dataset.

Parameters  
**idx** (*int*) – The index of the item to retrieve.

Returns  
A tuple containing the landmarks and target for the item.

Return type  
tuple

\_\_init\_\_(*metadata\_df=None*, *transform=None*, *max\_seq\_length=32*, *augment=False*, *augmentation\_threshold=0.1*, *enableDropout=True*)[¶](#data.dataset.ASL_DATASET.__init__ "Permalink to this definition")  
Initialize the ASL dataset.

This method initializes the dataset and loads the metadata necessary for
the dataset processing. If no metadata is provided, it will load the
default processed dataset. It also sets the transformation functions,
data augmentation parameters, and maximum sequence length.

Args:  
metadata\_df (pd.DataFrame, optional): A dataframe containing the
metadata for the dataset. Defaults to None. transform (callable,
optional): A function/transform to apply to the data. Defaults to None.
max\_seq\_length (int, optional): The maximum sequence length for the
data. Defaults to INPUT\_SIZE. augment (bool, optional): Whether to
apply data augmentation. Defaults to False. augmentation\_threshold
(float, optional): Probability of augmentation happening. Only if
augment == True. Defaults to 0.1. enableDropout (bool, optional):
Whether to enable the frame dropout augmentation. Defaults to True.

Functionality:  
Initializes the dataset with necessary configurations and loads the
data.

Parameters  
-   **metadata\_df** (*pd.DataFrame,* *optional*) – A dataframe
    containing the metadata for the dataset.

-   **transform** (*callable,* *optional*) – A function/transform to
    apply to the data.

-   **max\_seq\_length** (*int*) – The maximum sequence length for the
    data.

-   **augment** (*bool*) – Whether to apply data augmentation.

-   **augmentation\_threshold** (*float*) – Probability of augmentation
    happening. Only if augment == True.

-   **enableDropout** (*bool*) – Whether to enable the frame dropout
    augmentation.

\_\_len\_\_()[¶](#data.dataset.ASL_DATASET.__len__ "Permalink to this definition")  
Get the length of the dataset.

This method returns the total number of data samples present in the
dataset. It’s an implementation of the special method \_\_len\_\_ in
Python, providing a way to use the Python built-in function len() on the
dataset object.

Functionality:  
Get the length of the dataset.

Returns:  
int: The length of the dataset.

Returns  
The length of the dataset.

Return type  
int

\_\_repr\_\_()[¶](#data.dataset.ASL_DATASET.__repr__ "Permalink to this definition")  
Return a string representation of the ASL dataset.

This method returns a string that provides an overview of the dataset,
including the number of participants and total data samples. It’s an
implementation of the special method \_\_repr\_\_ in Python, providing a
human-readable representation of the dataset object.

Returns:  
str: A string representation of the dataset.

Functionality:  
Return a string representation of the dataset.

Returns  
A string representation of the dataset.

Return type  
str

\_\_weakref\_\_[¶](#data.dataset.ASL_DATASET.__weakref__ "Permalink to this definition")  
list of weak references to the object (if defined)

load\_data()[¶](#data.dataset.ASL_DATASET.load_data "Permalink to this definition")  
Load the data for the ASL dataset.

This method loads the actual ASL data based on the metadata provided
during initialization. If no metadata was provided, it loads the default
processed data. It generates absolute paths to locate landmark files,
and stores individual metadata lists for easy access during data
retrieval.

Functionality:  
Loads the data for the dataset.

Return type  
None

</div>

</div>

<div id="module-dl_utils" class="section">

## Data Utilities[¶](#module-dl_utils "Permalink to this headline")

<div id="deep-learning-utils" class="section">

### Deep Learning Utils[¶](#deep-learning-utils "Permalink to this headline")

This module provides a set of helper functions that abstract away
specific details of different deep learning frameworks (such as
TensorFlow and PyTorch). These functions allow the main code to run in a
framework-agnostic manner, thus improving code portability and
flexibility.

*class *dl\_utils.DatasetWithLen(*tf\_dataset*, *length*)[¶](#dl_utils.DatasetWithLen "Permalink to this definition")  
The DatasetWithLen class serves as a wrapper around TensorFlow’s Dataset
object. Its primary purpose is to add a length method to the TensorFlow
Dataset. This is useful in contexts where it’s necessary to know the
number of batches that a DataLoader will create from a dataset, which is
a common requirement in many machine learning training loops. It also
provides an iterator over the dataset, which facilitates traversing the
dataset for operations such as batch creation.

For instance, this might be used in conjunction with a progress bar
during training to display the total number of batches. Since
TensorFlow’s Dataset objects don’t inherently have a \_\_len\_\_ method,
this wrapper class provides that functionality, augmenting the dataset
with additional features that facilitate the training process.

Args:  
tf\_dataset: The TensorFlow dataset to be wrapped. length: The length of
the dataset.

Functionality:  
Provides a length method and an iterator for a TensorFlow dataset.

Return type  
DatasetWithLen object

Parameters  
-   **tf\_dataset** – The TensorFlow dataset to be wrapped.

-   **length** – The length of the dataset.

\_\_init\_\_(*tf\_dataset*, *length*)[¶](#dl_utils.DatasetWithLen.__init__ "Permalink to this definition")  

\_\_iter\_\_()[¶](#dl_utils.DatasetWithLen.__iter__ "Permalink to this definition")  
Returns an iterator for the dataset.

Returns  
iterator for the dataset

\_\_len\_\_()[¶](#dl_utils.DatasetWithLen.__len__ "Permalink to this definition")  
Returns the length of the dataset.

Returns  
length of the dataset

\_\_weakref\_\_[¶](#dl_utils.DatasetWithLen.__weakref__ "Permalink to this definition")  
list of weak references to the object (if defined)

&nbsp;

dl\_utils.get\_PT\_Dataset(*dataloader*)[¶](#dl_utils.get_PT_Dataset "Permalink to this definition")  
Retrieve the underlying dataset from a PyTorch data loader.

Parameters  
**dataloader** – DataLoader object.

Returns  
Dataset object.

&nbsp;

dl\_utils.get\_TF\_Dataset(*dataloader*)[¶](#dl_utils.get_TF_Dataset "Permalink to this definition")  
Retrieve the underlying dataset from a TensorFlow data loader.

Parameters  
**dataloader** – DatasetWithLen object.

Returns  
Dataset object.

&nbsp;

dl\_utils.get\_dataloader(*dataset*, *batch\_size=128*, *shuffle=True*, *dl\_framework='pytorch'*, *num\_workers=8*)[¶](#dl_utils.get_dataloader "Permalink to this definition")  
The get\_dataloader function is responsible for creating a DataLoader
object given a dataset and a few other parameters. A DataLoader is an
essential component in machine learning projects as it controls how data
is fed into the model during training. However, different deep learning
frameworks have their own ways of creating and handling DataLoader
objects.

To improve the portability and reusability of the code, this function
abstracts away these specifics, allowing the user to create a DataLoader
object without having to worry about the details of the underlying
framework (TensorFlow or PyTorch). This approach can save development
time and reduce the risk of bugs or errors.

Args:  
dataset: The dataset to be loaded. batch\_size: The size of the batches
that the DataLoader should create. shuffle: Whether to shuffle the data
before creating batches. dl\_framework: The name of the deep learning
framework. num\_workers: The number of worker threads to use for loading
data.

Functionality:  
Creates and returns a DataLoader object that is compatible with the
specified deep learning framework.

Return type  
DataLoader or DatasetWithLen object

Parameters  
-   **dataset** – The dataset to be loaded.

-   **batch\_size** – The size of the batches that the DataLoader should
    create.

-   **shuffle** – Whether to shuffle the data before creating batches.

-   **dl\_framework** – The name of the deep learning framework.

-   **num\_workers** – The number of worker threads to use for loading
    data.

&nbsp;

dl\_utils.get\_dataset(*dataloader*)[¶](#dl_utils.get_dataset "Permalink to this definition")  
The get\_dataset function is an interface to extract the underlying
dataset from a dataloader, irrespective of the deep learning framework
being used, i.e., TensorFlow or PyTorch. The versatility of this
function makes it integral to any pipeline designed to be flexible
across both TensorFlow and PyTorch frameworks.

Given a dataloader object, this function first determines the deep
learning framework currently in use by referring to the DL\_FRAMEWORK
config parameter variable. If the framework is TensorFlow, it invokes
the get\_TF\_Dataset function to retrieve the dataset. Alternatively, if
PyTorch is being used, the get\_PT\_Dataset function is called. This
abstracts away the intricacies of handling different deep learning
frameworks, thereby simplifying the process of working with datasets
across TensorFlow and PyTorch.

Args:  
dataloader: DataLoader from PyTorch or DatasetWithLen from TensorFlow.

Functionality:  
Extracts the underlying dataset from a dataloader, be it from PyTorch or
TensorFlow.

Return type  
Dataset object

Parameters  
**dataloader** – DataLoader in case of PyTorch and DatasetWithLen in
case of TensorFlow.

&nbsp;

dl\_utils.get\_model\_params(*model\_name*)[¶](#dl_utils.get_model_params "Permalink to this definition")  
The get\_model\_params function is a utility function that serves to
abstract away the details of reading model configurations from a YAML
file. In a machine learning project, it is common to have numerous
models, each with its own set of hyperparameters. These hyperparameters
can be stored in a YAML file for easy access and modification.

This function reads the configuration file and retrieves the specific
parameters associated with the given model. The configurations are
stored in a dictionary which is then returned. This aids in maintaining
a cleaner, more organized codebase and simplifies the process of
updating or modifying model parameters.

Args:  
model\_name: Name of the model whose parameters are to be retrieved.

Functionality:  
Reads a YAML file and retrieves the model parameters as a dictionary.

Return type  
dict

Parameters  
**model\_name** – Name of the model whose parameters are to be
retrieved.

&nbsp;

dl\_utils.log\_metrics(*phase*, *loss*, *acc*, *epoch*, *lr*, *writer*)[¶](#dl_utils.log_metrics "Permalink to this definition")  
Helper function to log metrics to TensorBoard.

Parameters  
-   **phase** – String, the phase of the process (‘train’ or
    ‘validation’).

-   **loss** – Float, the current loss value.

-   **acc** – Float, the current accuracy value.

-   **epoch** – Integer, the current epoch number.

-   **lr** – Float, the current learning rate.

-   **writer** – TensorBoard writer object.

&nbsp;

dl\_utils.to\_PT\_DataLoader(*dataset*, *batch\_size=128*, *shuffle=True*, *num\_workers=8*)[¶](#dl_utils.to_PT_DataLoader "Permalink to this definition")  
This function is the PyTorch counterpart to ‘to\_TF\_DataLoader’. It
converts a given dataset into a PyTorch DataLoader. The purpose of this
function is to streamline the creation of PyTorch DataLoaders, allowing
for easy utilization in a PyTorch training or inference pipeline.

The PyTorch DataLoader handles the process of drawing batches of data
from a dataset, which is essential when training models. This function
further extends this functionality by implementing data shuffling and
utilizing multiple worker threads for asynchronous data loading, thereby
optimizing the data loading process during model training.

Args:  
dataset: The dataset to be loaded. batch\_size: The size of each batch
the DataLoader will return. shuffle: Whether the data should be shuffled
before batching. num\_workers: The number of worker threads to use for
data loading.

Functionality:  
Converts a given dataset into a PyTorch DataLoader.

Return type  
DataLoader object

Parameters  
-   **dataset** – The dataset to be loaded.

-   **batch\_size** – The size of each batch the DataLoader will return.

-   **shuffle** – Whether the data should be shuffled before batching.

-   **num\_workers** – The number of worker threads to use for data
    loading.

&nbsp;

dl\_utils.to\_TF\_DataLoader(*dataset*, *batch\_size=128*, *shuffle=True*)[¶](#dl_utils.to_TF_DataLoader "Permalink to this definition")  
This function takes in a dataset and converts it into a TensorFlow
DataLoader. Its purpose is to provide a streamlined method to generate
DataLoaders that can be utilized in a TensorFlow training or inference
pipeline. It not only ensures the dataset is in a format that can be
ingested by TensorFlow’s pipeline, but also implements optional
shuffling of data, which is a common practice in model training to
ensure random distribution of data across batches.

This function first checks whether the data is already in a tensor
format, if not it converts the data to a tensor. Next, it either
shuffles the dataset or keeps it as is, based on the ‘shuffle’ flag.
Lastly, it prepares the TensorFlow DataLoader by batching the dataset
and applying an automatic optimization strategy for the number of
parallel calls in mapping functions.

Args:  
dataset: The dataset to be loaded. batch\_size: The size of each batch
the DataLoader will return. shuffle: Whether the data should be shuffled
before batching.

Functionality:  
Converts a given dataset into a TensorFlow DataLoader.

Return type  
DatasetWithLen object

Parameters  
-   **dataset** – The dataset to be loaded.

-   **batch\_size** – The size of each batch the DataLoader will return.

-   **shuffle** – Whether the data should be shuffled before batching.

</div>

</div>

<div id="module-trainer" class="section">

## Model Training[¶](#module-trainer "Permalink to this headline")

<div id="generic-trainer-description" class="section">

### Generic trainer description[¶](#generic-trainer-description "Permalink to this headline")

Trainer module handles the training, validation, and testing of
framework-agnostic deep learning models.

The Trainer class handles the complete lifecycle of model training
including setup, execution of training epochs, validation and testing,
early stopping, and result logging.

The class uses configurable parameters for defining training settings
like early stopping and batch size, and it supports adding custom
callback functions to be executed at the end of each epoch. This makes
the trainer class flexible and adaptable for various types of deep
learning models and tasks.

Attributes: model\_name (str): The name of the model to be trained.
params (dict): The parameters required for the model. model (model
object): The model object built using the given model name and
parameters. train\_loader, valid\_loader, test\_loader (DataLoader
objects): PyTorch dataloaders for training, validation, and testing
datasets. patience (int): The number of epochs to wait before stopping
training when the validation loss is no longer improving.
best\_val\_metric (float): The best validation metric recorded.
patience\_counter (int): A counter that keeps track of the number of
epochs since the validation loss last improved. model\_class (str): The
class name of the model. train\_start\_time (str): The starting time of
the training process. writer (SummaryWriter object): TensorBoard’s
SummaryWriter to log metrics for visualization. checkpoint\_path (str):
The path where the best model checkpoints will be saved during training.
epoch (int): The current epoch number. callbacks (list): A list of
callback functions to be called at the end of each epoch.

Methods: train(n\_epochs): Trains the model for a specified number of
epochs. evaluate(): Evaluates the model on the validation set. test():
Tests the model on the test set. add\_callback(callback): Adds a
callback function to the list of functions to be called at the end of
each epoch.

*class *trainer.Trainer(*modelname='YetAnotherTransformerClassifier'*, *dataset=&lt;class 'data.dataset.ASL\_DATASET'&gt;*, *patience=10*, *enableAugmentationDropout=True*, *augmentation\_threshold=0.35*)[¶](#trainer.Trainer "Permalink to this definition")  
A trainer class which acts as a control hub for the model lifecycle,
including initial setup, executing training epochs, performing
validation and testing, implementing early stopping, and logging
results. The module has been designed to be agnostic to the specific
deep learning framework, enhancing its versatility across various
projects.

\_\_init\_\_(*modelname='YetAnotherTransformerClassifier'*, *dataset=&lt;class 'data.dataset.ASL\_DATASET'&gt;*, *patience=10*, *enableAugmentationDropout=True*, *augmentation\_threshold=0.35*)[¶](#trainer.Trainer.__init__ "Permalink to this definition")  
Initializes the Trainer class with the specified parameters.

This method initializes various components needed for the training
process. This includes the model specified by the model name, the
dataset with optional data augmentation and dropout, data loaders for
the training, validation, and test sets, a SummaryWriter for logging,
and a path for saving model checkpoints.

1.  The method first retrieves the specified model and its parameters.

2.  It then initializes the dataset and the data loaders.

3.  It sets up metrics for early stopping and a writer for logging.

4.  Finally, it prepares a directory for saving model checkpoints.

Args:  
modelname (str): The name of the model to be used for training. dataset
(Dataset): The dataset to be used. patience (int): The number of epochs
with no improvement after which training will be stopped.
enableAugmentationDropout (bool): If True, enable data augmentation
dropout. augmentation\_threshold (float): The threshold for data
augmentation.

Functionality:  
This method initializes various components, such as the model, dataset,
data loaders, logging writer, and checkpoint path, required for the
training process.

Parameters  
-   **modelname** (*str*) – The name of the model for training.

-   **dataset** (*Dataset*) – The dataset for training.

-   **patience** (*int*) – The number of epochs with no improvement
    after which training will be stopped.

-   **enableAugmentationDropout** (*bool*) – If True, enable data
    augmentation dropout.

-   **augmentation\_threshold** (*float*) – The threshold for data
    augmentation.

Return type  
None

<div class="admonition note">

Note

This method only initializes the Trainer class. The actual training is
done by calling the train() method.

</div>

<div class="admonition warning">

Warning

Make sure the specified model name corresponds to an actual model in
your project’s models directory.

</div>

\_\_weakref\_\_[¶](#trainer.Trainer.__weakref__ "Permalink to this definition")  
list of weak references to the object (if defined)

add\_callback(*callback*)[¶](#trainer.Trainer.add_callback "Permalink to this definition")  
Adds a callback to the Trainer.

This method simply appends a callback function to the list of callbacks
stored by the Trainer instance. These callbacks are called at the end of
each training epoch.

Functionality:  
It allows the addition of custom callbacks to the training process,
enhancing its flexibility.

Parameters  
**callback** (*Callable*) – The callback function to be added.

Returns  
None

Return type  
None

<div class="admonition warning">

Warning

The callback function must be callable and should not modify the
training process.

</div>

evaluate()[¶](#trainer.Trainer.evaluate "Permalink to this definition")  
Evaluates the model on the validation set.

This method sets the model to evaluation mode and loops over the
validation dataset, computing the loss and accuracy for each batch. It
then averages these metrics and logs them. This process provides an
unbiased estimate of the model’s performance on new data during
training.

Functionality:  
It manages the evaluation of the model on the validation set, handling
batch-wise loss computation and accuracy assessment.

Returns  
Average validation loss and accuracy

Return type  
Tuple\[float, float\]

<div class="admonition warning">

Warning

Ensure the model is in evaluation mode to correctly compute the
validation metrics.

</div>

test()[¶](#trainer.Trainer.test "Permalink to this definition")  
Tests the model on the test set.

This method loads the best saved model, sets it to evaluation mode, and
then loops over the test dataset, computing the loss, accuracy, and
predictions for each batch. It then averages the loss and accuracy and
logs them. It also collects all the model’s predictions and their
corresponding labels.

Functionality:  
It manages the testing of the model on the test set, handling batch-wise
loss computation, accuracy assessment, and prediction generation.

Returns  
List of all predictions and their corresponding labels

Return type  
Tuple\[List, List\]

train(*n\_epochs=50*)[¶](#trainer.Trainer.train "Permalink to this definition")  
Trains the model for a specified number of epochs.

This method manages the main training loop of the model. For each epoch,
it performs several steps. It first puts the model into training mode
and loops over the training dataset, calculating the loss and accuracy
for each batch and optimizing the model parameters. It logs these
metrics and updates a progress bar. At the end of each epoch, it
evaluates the model on the validation set and checks whether early
stopping criteria have been met. If the early stopping metric has
improved, it saves the current model and its parameters. If not, it
increments a counter and potentially stops training if the counter
exceeds the allowed patience. Finally, it steps the learning rate
scheduler and calls any registered callbacks.

1.  The method first puts the model into training mode and initializes
    some lists and counters.

2.  Then it enters the main loop over the training data, updating the
    model and logging metrics.

3.  It evaluates the model on the validation set and checks the early
    stopping criteria.

4.  If the criteria are met, it saves the model and its parameters; if
    not, it increments a patience counter.

5.  It steps the learning rate scheduler and calls any callbacks.

Args:  
n\_epochs (int): The number of epochs for which the model should be
trained.

Functionality:  
This method coordinates the training of the model over a series of
epochs, handling batch-wise loss computation, backpropagation,
optimization, validation, early stopping, and model checkpoint saving.

Parameters  
**n\_epochs** (*int*) – Number of epochs for training.

Returns  
None

Return type  
None

<div class="admonition note">

Note

This method modifies the state of the model and its optimizer, as well
as various attributes of the Trainer instance itself.

</div>

<div class="admonition warning">

Warning

If you set the patience value too low in the constructor, the model
might stop training prematurely.

</div>

</div>

</div>

<div id="module-visualizations" class="section">

## Data Visualizations[¶](#module-visualizations "Permalink to this headline")

visualizations.visualize\_data\_distribution(*dataset*)[¶](#visualizations.visualize_data_distribution "Permalink to this definition")  
Visualize the distribution of data in terms of the number of samples and
average sequence length per class.

This function generates two bar charts: one showing the number of
samples per class, and the other showing the average sequence length per
class.

Parameters  
**dataset** (*ASL\_Dataset*) – The ASL dataset to load data from.

&nbsp;

visualizations.visualize\_target\_sign(*dataset*, *target\_sign*, *n\_samples=6*)[¶](#visualizations.visualize_target_sign "Permalink to this definition")  
Visualize n\_samples instances of a given target sign from the dataset.

This function generates a visual representation of the landmarks for
each sample belonging to the specified target\_sign.

Args:  
dataset (ASL\_Dataset): The ASL dataset to load data from. target\_sign
(int): The target sign to visualize. n\_samples (int, optional): The
number of samples to visualize. Defaults to 6.

Returns:  
matplotlib.animation.FuncAnimation: A matplotlib animation object
displaying the landmarks for each frame.

Parameters  
-   **dataset** (*ASL\_Dataset*) – The ASL dataset to load data from.

-   **target\_sign** (*int*) – The target sign to visualize.

-   **n\_samples** (*int,* *optional*) – The number of samples to
    visualize, defaults to 6.

Returns  
A matplotlib animation object displaying the landmarks for each frame.

Return type  
matplotlib.animation.FuncAnimation

</div>

<div id="module-models.pytorch.models" class="section">

## Pytorch Models[¶](#module-models.pytorch.models "Permalink to this headline")

This kodule defines a PyTorch BaseModel providing a basic framework for
learning and validating from Trainer module, from which other pytorch
models are inherited. This module includes several model classes that
build upon the PyTorch’s nn.Module for constructing pytorch LSTM or
Transformer based models:

| Class                           | Description                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TransformerSequenceClassifier   | This is a transformer-based sequence classification model. The class constructs a transformer encoder based on user-defined parameters or default settings. The forward method first checks and reshapes the input, then passes it through the transformer layers. It then pools the sequence by taking the mean over the time dimension, and finally applies the output layer to generate the class predictions. |
| TransformerPredictor            | A TransformerPredictor model that extends the Pytorch BaseModel. This class wraps TransformerSequenceClassifier model and provides functionality to use it for making predictions.                                                                                                                                                                                                                                |
| MultiHeadSelfAttention          | This class applies a multi-head attention mechanism. It has options for causal masking and layer normalization. The input is expected to have dimensions \[batch\_size, seq\_len, features\].                                                                                                                                                                                                                     |
| TransformerBlock                | This class represents a single block of a transformer architecture, including multi-head self-attention and a feed-forward neural network, both with optional layer normalization and dropout. The input is expected to have dimensions \[batch\_size, seq\_len, features\].                                                                                                                                      |
| YetAnotherTransformerClassifier | This class constructs a transformer-based classifier with a specified number of TransformerBlock instances. The output of the model is a tensor of logits with dimensions \[batch\_size, num\_classes\].                                                                                                                                                                                                          |
| YetAnotherTransformer           | This class is a wrapper for YetAnotherTransformerClassifier which includes learning rate, optimizer, and learning rate scheduler settings. It extends from the BaseModel class.                                                                                                                                                                                                                                   |
| YetAnotherEnsemble              | This class constructs an ensemble of YetAnotherTransformerClassifier instances, where the outputs are concatenated and passed through a fully connected layer. This class also extends from the BaseModel class and includes learning rate, optimizer, and learning rate scheduler settings.                                                                                                                      |

Model Classes[¶](#id1 "Permalink to this table")

*class *models.pytorch.models.BaseModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.BaseModel "Permalink to this definition")  
A BaseModel that extends the nn.Module from PyTorch.

Functionality: \#. The class initializes with a given learning rate and
number of classes. \#. It sets up the loss criterion, accuracy metric,
and default states for optimizer and scheduler. \#. It defines an
abstract method ‘forward’ which should be implemented in the
subclass. \#. It also defines various utility functions like calculating
accuracy, training, validation and testing steps, scheduler stepping,
and model checkpointing.

Args:  
learning\_rate (float): The initial learning rate for optimizer.
n\_classes (int): The number of classes for classification.

Parameters  
-   **learning\_rate** (*float*) – The initial learning rate for
    optimizer.

-   **n\_classes** (*int*) – The number of classes for classification.

Returns  
None

Return type  
None

<div class="admonition note">

Note

The class does not directly initialize the optimizer and scheduler. They
should be initialized in the subclass if needed.

</div>

<div class="admonition warning">

Warning

The ‘forward’ function must be implemented in the subclass, else it will
raise a NotImplementedError.

</div>

\_\_init\_\_(*learning\_rate*, *n\_classes=250*)[¶](#models.pytorch.models.BaseModel.__init__ "Permalink to this definition")  

calculate\_accuracy(*y\_hat*, *y*)[¶](#models.pytorch.models.BaseModel.calculate_accuracy "Permalink to this definition")  
Calculates the accuracy of the model’s prediction.

Parameters  
-   **y\_hat** (*Tensor*) – The predicted output from the model.

-   **y** (*Tensor*) – The ground truth or actual labels.

Returns  
The calculated accuracy.

Return type  
Tensor

eval\_mode()[¶](#models.pytorch.models.BaseModel.eval_mode "Permalink to this definition")  
Sets the model to evaluation mode.

forward(*x*)[¶](#models.pytorch.models.BaseModel.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

get\_lr()[¶](#models.pytorch.models.BaseModel.get_lr "Permalink to this definition")  
Gets the current learning rate of the model.

Returns  
The current learning rate.

Return type  
float

load\_checkpoint(*filepath*)[¶](#models.pytorch.models.BaseModel.load_checkpoint "Permalink to this definition")  
Loads the model and optimizer states from a checkpoint.

Parameters  
**filepath** (*str*) – The file path where to load the model checkpoint
from.

optimize()[¶](#models.pytorch.models.BaseModel.optimize "Permalink to this definition")  
Steps the optimizer and sets the gradients of all optimized
`torch.Tensor` s to zero.

save\_checkpoint(*filepath*)[¶](#models.pytorch.models.BaseModel.save_checkpoint "Permalink to this definition")  
Saves the model and optimizer states to a checkpoint.

Parameters  
**filepath** (*str*) – The file path where to save the model checkpoint.

step\_scheduler()[¶](#models.pytorch.models.BaseModel.step_scheduler "Permalink to this definition")  
Steps the learning rate scheduler, adjusting the optimizer’s learning
rate as necessary.

test\_step(*batch*)[¶](#models.pytorch.models.BaseModel.test_step "Permalink to this definition")  
Performs a test step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss, accuracy, and model predictions.

Return type  
tuple

train\_mode()[¶](#models.pytorch.models.BaseModel.train_mode "Permalink to this definition")  
Sets the model to training mode.

training\_step(*batch*)[¶](#models.pytorch.models.BaseModel.training_step "Permalink to this definition")  
Performs a training step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss and accuracy.

Return type  
tuple

validation\_step(*batch*)[¶](#models.pytorch.models.BaseModel.validation_step "Permalink to this definition")  
Performs a validation step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss and accuracy.

Return type  
tuple

&nbsp;

*class *models.pytorch.models.CVTransferLearningModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.CVTransferLearningModel "Permalink to this definition")  
A CVTransferLearningModel that extends the Pytorch BaseModel.

This class applies transfer learning for computer vision tasks using
pretrained models. It also provides a forward method to pass an input
through the model.

learning\_ratefloat  
The learning rate for the optimizer.

modelnn.Module  
The base model for transfer learning.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.CVTransferLearningModel.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.CVTransferLearningModel.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.HybridEnsembleModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.HybridEnsembleModel "Permalink to this definition")  
A HybridEnsembleModel that extends the Pytorch BaseModel.

This class creates an ensemble of LSTM and Transformer models and
provides functionality to use the ensemble for making predictions.

learning\_ratefloat  
The learning rate for the optimizer.

lstmsnn.ModuleList  
The list of LSTM models.

modelsnn.ModuleList  
The list of Transformer models.

fcnn.Linear  
The final fully-connected layer.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.HybridEnsembleModel.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.HybridEnsembleModel.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.HybridModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.HybridModel "Permalink to this definition")  
A HybridModel that extends the Pytorch BaseModel.

This class combines the LSTMClassifier and TransformerSequenceClassifier
models and provides functionality to use the combined model for making
predictions.

lstmLSTMClassifier  
The LSTM classifier used for making predictions.

transformerTransformerSequenceClassifier  
The transformer sequence classifier used for making predictions.

fcnn.Linear  
The final fully-connected layer.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.HybridModel.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.HybridModel.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.LSTMClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.LSTMClassifier "Permalink to this definition")  
A LSTM-based Sequence Classifier. This class utilizes a LSTM network for
sequence classification tasks.

DEFAULTSdict  
Default settings for the LSTM and classifier. These can be overridden by
passing values in the constructor.

lstmnn.LSTM  
The LSTM network used for processing the input sequence.

dropoutnn.Dropout  
The dropout layer applied after LSTM network.

output\_layernn.Linear  
The output layer used to generate class predictions.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.LSTMClassifier.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.LSTMClassifier.forward "Permalink to this definition")  
Forward pass through the model

&nbsp;

*class *models.pytorch.models.LSTMPredictor(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.LSTMPredictor "Permalink to this definition")  
A LSTMPredictor model that extends the Pytorch BaseModel.

This class wraps the LSTMClassifier model and provides functionality to
use it for making predictions.

learning\_ratefloat  
The learning rate for the optimizer.

modelLSTMClassifier  
The LSTM classifier used for making predictions.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.LSTMPredictor.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.LSTMPredictor.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.MultiHeadSelfAttention(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.MultiHeadSelfAttention "Permalink to this definition")  
A MultiHeadSelfAttention module that extends the nn.Module from PyTorch.

Functionality: \#. The class initializes with a given dimension size,
number of attention heads, dropout rate, layer normalization and
causality. \#. It sets up the multihead attention module and layer
normalization. \#. It also defines a forward method that applies the
multihead attention, causal masking if requested, and layer
normalization if requested.

multihead\_attnnn.MultiheadAttention  
The multihead attention module.

layer\_normnn.LayerNorm or None  
The layer normalization module. If it is not applied, set to None.

causalbool  
If True, applies causal masking.

forward(x)  
Performs a forward pass through the model.

Args:  
dim (int): The dimension size of the input data. num\_heads (int): The
number of attention heads. dropout (float): The dropout rate.
layer\_norm (bool): Whether to apply layer normalization. causal (bool):
Whether to apply causal masking.

Returns: None

\_\_init\_\_(*dim*, *num\_heads=8*, *dropout=0.1*, *layer\_norm=True*, *causal=True*)[¶](#models.pytorch.models.MultiHeadSelfAttention.__init__ "Permalink to this definition")  

&nbsp;

*class *models.pytorch.models.TransformerBlock(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.TransformerBlock "Permalink to this definition")  
A TransformerBlock module that extends the nn.Module from PyTorch.

Functionality: \#. The class initializes with a given dimension size,
number of attention heads, expansion factor, attention dropout rate, and
dropout rate. \#. It sets up the multihead self-attention module, layer
normalization and feed-forward network. \#. It also defines a forward
method that applies the multihead self-attention, dropout, layer
normalization and feed-forward network.

norm1, norm2, norm3nn.LayerNorm  
The layer normalization modules.

attnMultiHeadSelfAttention  
The multihead self-attention module.

feed\_forwardnn.Sequential  
The feed-forward network.

dropoutnn.Dropout  
The dropout module.

forward(x)  
Performs a forward pass through the model.

Args:  
dim (int): The dimension size of the input data. num\_heads (int): The
number of attention heads. expansion\_factor (int): The expansion factor
for the hidden layer size in the feed-forward network. attn\_dropout
(float): The dropout rate for the attention module. drop\_rate (float):
The dropout rate for the module.

Returns: None

\_\_init\_\_(*dim=192*, *num\_heads=4*, *expansion\_factor=4*, *attn\_dropout=0.2*, *drop\_rate=0.2*)[¶](#models.pytorch.models.TransformerBlock.__init__ "Permalink to this definition")  

&nbsp;

*class *models.pytorch.models.TransformerEnsemble(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.TransformerEnsemble "Permalink to this definition")  
A TransformerEnsemble that extends the Pytorch BaseModel.

This class creates an ensemble of TransformerSequenceClassifier models
and provides functionality to use the ensemble for making predictions.

learning\_ratefloat  
The learning rate for the optimizer.

modelsnn.ModuleList  
The list of transformer sequence classifiers used for making
predictions.

fcnn.Linear  
The final fully-connected layer.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.TransformerEnsemble.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.TransformerEnsemble.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.TransformerPredictor(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.TransformerPredictor "Permalink to this definition")  
A TransformerPredictor model that extends the Pytorch BaseModel.

This class wraps the TransformerSequenceClassifier model and provides
functionality to use it for making predictions.

learning\_ratefloat  
The learning rate for the optimizer.

modelTransformerSequenceClassifier  
The transformer sequence classifier used for making predictions.

optimizertorch.optim.Adam  
The optimizer used for updating the model parameters.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler used for adapting the learning rate during
training.

forward(x)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.TransformerPredictor.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.TransformerPredictor.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.TransformerSequenceClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.TransformerSequenceClassifier "Permalink to this definition")  
A Transformer-based Sequence Classifier. This class utilizes a
transformer encoder to process the input sequence.

The transformer encoder consists of a stack of N transformer layers that
are applied to the input sequence. The output sequence from the
transformer encoder is then passed through a linear layer to generate
class predictions.

DEFAULTSdict  
Default settings for the transformer encoder and classifier. These can
be overridden by passing values in the constructor.

transformernn.TransformerEncoder  
The transformer encoder used to process the input sequence.

output\_layernn.Linear  
The output layer used to generate class predictions.

batch\_firstbool  
Whether the first dimension of the input tensor represents the batch
size.

forward(inputs)  
Performs a forward pass through the model.

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.TransformerSequenceClassifier.__init__ "Permalink to this definition")  

forward(*inputs*)[¶](#models.pytorch.models.TransformerSequenceClassifier.forward "Permalink to this definition")  
Forward pass through the model

&nbsp;

*class *models.pytorch.models.YetAnotherEnsemble(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.YetAnotherEnsemble "Permalink to this definition")  
A YetAnotherEnsemble model that extends the Pytorch BaseModel.

Functionality: \#. The class initializes with a set of parameters for
the YetAnotherTransformerClassifier. \#. It sets up an ensemble of
YetAnotherTransformerClassifier models, a fully connected layer, the
optimizer and the learning rate scheduler. \#. It also defines a forward
method that applies each YetAnotherTransformerClassifier model in the
ensemble, concatenates the outputs and applies the fully connected
layer.

Args:  
kwargs (dict): A dictionary containing the parameters for the
YetAnotherTransformerClassifier models, fully connected layer, optimizer
and learning rate scheduler.

Returns: None

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.YetAnotherEnsemble.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.YetAnotherEnsemble.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.YetAnotherTransformer(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.YetAnotherTransformer "Permalink to this definition")  
A YetAnotherTransformer model that extends the Pytorch BaseModel.

Functionality: \#. The class initializes with a set of parameters for
the YetAnotherTransformerClassifier. \#. It sets up the
YetAnotherTransformerClassifier model, the optimizer and the learning
rate scheduler. \#. It also defines a forward method that applies the
YetAnotherTransformerClassifier model.

learning\_ratefloat  
The learning rate for the optimizer.

modelYetAnotherTransformerClassifier  
The YetAnotherTransformerClassifier model.

optimizertorch.optim.AdamW  
The AdamW optimizer.

schedulertorch.optim.lr\_scheduler.ExponentialLR  
The learning rate scheduler.

forward(x)  
Performs a forward pass through the model.

Args:  
kwargs (dict): A dictionary containing the parameters for the
YetAnotherTransformerClassifier, optimizer and learning rate scheduler.

Returns: None

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.YetAnotherTransformer.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.models.YetAnotherTransformer.forward "Permalink to this definition")  
The forward function for the BaseModel.

Parameters  
**x** (*Tensor*) – The inputs to the model.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.pytorch.models.YetAnotherTransformerClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.models.YetAnotherTransformerClassifier "Permalink to this definition")  
A YetAnotherTransformerClassifier module that extends the nn.Module from
PyTorch.

Functionality: \#. The class initializes with a set of parameters for
the transformer blocks. \#. It sets up the transformer blocks and the
output layer. \#. It also defines a forward method that applies the
transformer blocks, takes the mean over the time dimension of the
transformed sequence, and applies the output layer.

DEFAULTSdict  
The default settings for the transformer.

settingsdict  
The settings for the transformer, with any user-provided values
overriding the defaults.

transformernn.ModuleList  
The list of transformer blocks.

output\_layernn.Linear  
The output layer.

forward(inputs)  
Performs a forward pass through the model.

Args:  
kwargs (dict): A dictionary containing the parameters for the
transformer blocks.

Returns: None

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.models.YetAnotherTransformerClassifier.__init__ "Permalink to this definition")  

forward(*inputs*)[¶](#models.pytorch.models.YetAnotherTransformerClassifier.forward "Permalink to this definition")  
Forward pass through the model

</div>

<div id="module-models.tensorflow.models" class="section">

## Tensorflow Models[¶](#module-models.tensorflow.models "Permalink to this headline")

This kodule defines a PyTorch BaseModel providing a basic framework for
learning and validating from Trainer

*class *models.tensorflow.models.BaseModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.BaseModel "Permalink to this definition")  
A BaseModel that extends the tf.keras.Model.

Functionality: \#. The class initializes with a given learning rate. \#.
It sets up the loss criterion, accuracy metric, and default states for
optimizer and scheduler. \#. It defines an abstract method ‘call’ which
should be implemented in the subclass. \#. It also defines various
utility functions like calculating accuracy, training, validation and
testing steps, scheduler stepping, and model checkpointing.

Args:  
learning\_rate (float): The initial learning rate for optimizer.

Parameters  
**learning\_rate** (*float*) – The initial learning rate for optimizer.

Returns  
None

Return type  
None

<div class="admonition note">

Note

The class does not directly initialize the optimizer and scheduler. They
should be initialized in the subclass if needed.

</div>

<div class="admonition warning">

Warning

The ‘call’ function must be implemented in the subclass, else it will
raise a NotImplementedError.

</div>

\_\_init\_\_(*learning\_rate*)[¶](#models.tensorflow.models.BaseModel.__init__ "Permalink to this definition")  

calculate\_accuracy(*y\_pred*, *y\_true*)[¶](#models.tensorflow.models.BaseModel.calculate_accuracy "Permalink to this definition")  
Calculates the accuracy of the model’s prediction.

Parameters  
-   **y\_pred** (*Tensor*) – The predicted output from the model.

-   **y\_true** (*Tensor*) – The ground truth or actual labels.

Returns  
The calculated accuracy.

Return type  
float

call(*inputs*, *training=False*)[¶](#models.tensorflow.models.BaseModel.call "Permalink to this definition")  
The call function for the BaseModel.

Parameters  
-   **inputs** (*Tensor*) – The inputs to the model.

-   **training** (*bool*) – A flag indicating whether the model is in
    training mode. Default is False.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

eval\_mode()[¶](#models.tensorflow.models.BaseModel.eval_mode "Permalink to this definition")  
Sets the model to evaluation mode.

get\_lr()[¶](#models.tensorflow.models.BaseModel.get_lr "Permalink to this definition")  
Gets the current learning rate of the model.

Returns  
The current learning rate.

Return type  
float

load\_checkpoint(*filepath*)[¶](#models.tensorflow.models.BaseModel.load_checkpoint "Permalink to this definition")  
Loads the model weights from a checkpoint.

Parameters  
**filepath** (*str*) – The file path where to load the model checkpoint
from.

optimize()[¶](#models.tensorflow.models.BaseModel.optimize "Permalink to this definition")  
Sets the model to training mode.

save\_checkpoint(*filepath*)[¶](#models.tensorflow.models.BaseModel.save_checkpoint "Permalink to this definition")  
Saves the model weights to a checkpoint.

Parameters  
**filepath** (*str*) – The file path where to save the model checkpoint.

step\_scheduler()[¶](#models.tensorflow.models.BaseModel.step_scheduler "Permalink to this definition")  
Adjusts the learning rate according to the learning rate scheduler.

test\_step(*batch*)[¶](#models.tensorflow.models.BaseModel.test_step "Permalink to this definition")  
Performs a test step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss, accuracy, and model predictions.

Return type  
tuple

train\_mode()[¶](#models.tensorflow.models.BaseModel.train_mode "Permalink to this definition")  
Sets the model to training mode.

training\_step(*batch*)[¶](#models.tensorflow.models.BaseModel.training_step "Permalink to this definition")  
Performs a training step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss and accuracy.

Return type  
tuple

validation\_step(*batch*)[¶](#models.tensorflow.models.BaseModel.validation_step "Permalink to this definition")  
Performs a validation step using the input batch data.

Parameters  
**batch** (*tuple*) – A tuple containing input data and labels.

Returns  
The calculated loss and accuracy.

Return type  
tuple

&nbsp;

*class *models.tensorflow.models.HybridModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.HybridModel "Permalink to this definition")  
\_\_init\_\_(*\*\*kwargs*)[¶](#models.tensorflow.models.HybridModel.__init__ "Permalink to this definition")  

call(*inputs*, *training=True*)[¶](#models.tensorflow.models.HybridModel.call "Permalink to this definition")  
The call function for the BaseModel.

Parameters  
-   **inputs** (*Tensor*) – The inputs to the model.

-   **training** (*bool*) – A flag indicating whether the model is in
    training mode. Default is False.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.tensorflow.models.LSTMClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.LSTMClassifier "Permalink to this definition")  
LSTM-based Sequence Classifier

\_\_init\_\_(*\*\*kwargs*)[¶](#models.tensorflow.models.LSTMClassifier.__init__ "Permalink to this definition")  

call(*inputs*)[¶](#models.tensorflow.models.LSTMClassifier.call "Permalink to this definition")  
Forward pass through the model

&nbsp;

*class *models.tensorflow.models.LSTMPredictor(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.LSTMPredictor "Permalink to this definition")  
\_\_init\_\_(*\*\*kwargs*)[¶](#models.tensorflow.models.LSTMPredictor.__init__ "Permalink to this definition")  

call(*inputs*)[¶](#models.tensorflow.models.LSTMPredictor.call "Permalink to this definition")  
The call function for the BaseModel.

Parameters  
-   **inputs** (*Tensor*) – The inputs to the model.

-   **training** (*bool*) – A flag indicating whether the model is in
    training mode. Default is False.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.tensorflow.models.TransformerEncoderLayer(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.TransformerEncoderLayer "Permalink to this definition")  
A Transformer Encoder layer as a subclass of tf.keras.layers.Layer.

Functionality: \#. The class first initializes with key parameters for
MultiHeadAttention and feedforward network. \#. Then it defines the key
components like multi-head attention, feedforward network, layer
normalization, and dropout. \#. In the call function, it takes input and
performs self-attention, followed by layer normalization and feedforward
operation.

Args:  
d\_model (int): The dimensionality of the input. n\_head (int): The
number of heads in the multi-head attention. dim\_feedforward (int): The
dimensionality of the feedforward network model. dropout (float): The
dropout value.

Parameters  
-   **d\_model** (*int*) – The dimensionality of the input.

-   **n\_head** (*int*) – The number of heads in the multi-head
    attention.

-   **dim\_feedforward** (*int*) – The dimensionality of the feedforward
    network model.

-   **dropout** (*float*) – The dropout value.

Returns  
None

Return type  
None

<div class="admonition note">

Note

The implementation is based on the “Attention is All You Need” paper.

</div>

<div class="admonition warning">

Warning

Ensure that the input dimension ‘d\_model’ is divisible by the number of
attention heads ‘n\_head’.

</div>

\_\_init\_\_(*d\_model*, *n\_head*, *dim\_feedforward*, *dropout*)[¶](#models.tensorflow.models.TransformerEncoderLayer.__init__ "Permalink to this definition")  

&nbsp;

*class *models.tensorflow.models.TransformerPredictor(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.TransformerPredictor "Permalink to this definition")  
A Transformer Predictor model that extends the BaseModel.

Functionality: \#. The class first initializes with the learning rate
and other parameters. \#. It then creates an instance of
TransformerSequenceClassifier. \#. It also sets up the learning rate
scheduler and the optimizer. \#. In the call function, it simply runs
the TransformerSequenceClassifier.

Args:  
kwargs (dict): A dictionary of arguments.

Parameters  
**kwargs** (*dict*) – A dictionary of arguments.

Returns  
None

Return type  
None

<div class="admonition note">

Note

The learning rate is set up with an exponential decay schedule.

</div>

<div class="admonition warning">

Warning

The learning rate and gamma for the decay schedule must be specified in
the ‘kwargs’.

</div>

\_\_init\_\_(*\*\*kwargs*)[¶](#models.tensorflow.models.TransformerPredictor.__init__ "Permalink to this definition")  

call(*inputs*, *training=True*)[¶](#models.tensorflow.models.TransformerPredictor.call "Permalink to this definition")  
The call function for the BaseModel.

Parameters  
-   **inputs** (*Tensor*) – The inputs to the model.

-   **training** (*bool*) – A flag indicating whether the model is in
    training mode. Default is False.

Returns  
None

<div class="admonition warning">

Warning

This function must be implemented in the subclass, else it raises a
NotImplementedError.

</div>

&nbsp;

*class *models.tensorflow.models.TransformerSequenceClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.tensorflow.models.TransformerSequenceClassifier "Permalink to this definition")  
A Transformer Sequence Classifier as a subclass of tf.keras.Model.

Functionality: \#. The class first initializes with default or provided
settings. \#. Then it defines the key components like the transformer
encoder layers and output layer. \#. In the call function, it takes
input and passes it through each transformer layer followed by
normalization and dense layer for final output.

Args:  
kwargs (dict): Any additional arguments. If not provided, defaults will
be used.

Parameters  
**kwargs** (*dict*) – Any additional arguments.

Returns  
None

Return type  
None

<div class="admonition note">

Note

The implementation is based on the “Attention is All You Need” paper.

</div>

<div class="admonition warning">

Warning

The inputs should have a shape of (batch\_size, seq\_length, height,
width), otherwise, a ValueError will be raised.

</div>

\_\_init\_\_(*\*\*kwargs*)[¶](#models.tensorflow.models.TransformerSequenceClassifier.__init__ "Permalink to this definition")  

</div>

<div id="module-models.pytorch.lightning_models" class="section">

## Torch Lightning Models[¶](#module-models.pytorch.lightning_models "Permalink to this headline")

*class *models.pytorch.lightning\_models.LightningBaseModel(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.lightning_models.LightningBaseModel "Permalink to this definition")  
\_\_init\_\_(*learning\_rate*, *n\_classes=250*)[¶](#models.pytorch.lightning_models.LightningBaseModel.__init__ "Permalink to this definition")  

configure\_optimizers()[¶](#models.pytorch.lightning_models.LightningBaseModel.configure_optimizers "Permalink to this definition")  
Choose what optimizers and learning-rate schedulers to use in your
optimization. Normally you’d need one. But in the case of GANs or
similar you might have multiple. Optimization with multiple optimizers
only works in the manual optimization mode.

Return:  
Any of these 6 options.

-   **Single optimizer**.

-   **List or Tuple** of optimizers.

-   **Two lists** - The first list has multiple optimizers, and the
    second has multiple LR schedulers (or multiple
    `lr_scheduler_config`).

-   **Dictionary**, with an `"optimizer"` key, and (optionally) a
    `"lr_scheduler"` key whose value is a single LR scheduler or
    `lr_scheduler_config`.

-   **None** - Fit will run without any optimizer.

The `lr_scheduler_config` is a dictionary which contains the scheduler
and its associated configuration. The default configuration is shown
below.

<div class="highlight-python notranslate">

<div class="highlight">

    lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": lr_scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
    }

</div>

</div>

When there are schedulers in which the `.step()` method is conditioned
on a value, such as the `torch.optim.lr_scheduler.ReduceLROnPlateau`
scheduler, Lightning requires that the `lr_scheduler_config` contains
the keyword `"monitor"` set to the metric name that the scheduler should
be conditioned on.

Metrics can be made available to monitor by simply logging it using
`self.log('metric_to_track', metric_val)` in your `LightningModule`.

Note:  
Some things to know:

-   Lightning calls `.backward()` and `.step()` automatically in case of
    automatic optimization.

-   If a learning rate scheduler is specified in
    `configure_optimizers()` with key `"interval"` (default “epoch”) in
    the scheduler configuration, Lightning will call the scheduler’s
    `.step()` method automatically in case of automatic optimization.

-   If you use 16-bit precision (`precision=16`), Lightning will
    automatically handle the optimizer.

-   If you use `torch.optim.LBFGS`, Lightning handles the closure
    function automatically for you.

-   If you use multiple optimizers, you will have to switch to ‘manual
    optimization’ mode and step them yourself.

-   If you need to control how often the optimizer steps, override the
    `optimizer_step()` hook.

forward(*x*)[¶](#models.pytorch.lightning_models.LightningBaseModel.forward "Permalink to this definition")  
Same as `torch.nn.Module.forward()`.

Args:  
[\*](#id1)args: Whatever you decide to pass into the forward method.
[\*\*](#id3)kwargs: Keyword arguments are also possible.

Return:  
Your model’s output

on\_test\_end() → None[¶](#models.pytorch.lightning_models.LightningBaseModel.on_test_end "Permalink to this definition")  
Called at the end of testing.

on\_train\_epoch\_end() → None[¶](#models.pytorch.lightning_models.LightningBaseModel.on_train_epoch_end "Permalink to this definition")  
Called in the training loop at the very end of the epoch.

To access all batch outputs at the end of the epoch, you can cache step
outputs as an attribute of the `LightningModule` and access them in this
hook:

<div class="highlight-python notranslate">

<div class="highlight">

    class MyLightningModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.training_step_outputs = []

        def training_step(self):
            loss = ...
            self.training_step_outputs.append(loss)
            return loss

        def on_train_epoch_end(self):
            # do something with all training_step outputs, for example:
            epoch_mean = torch.stack(self.training_step_outputs).mean()
            self.log("training_epoch_mean", epoch_mean)
            # free up the memory
            self.training_step_outputs.clear()

</div>

</div>

on\_validation\_end()[¶](#models.pytorch.lightning_models.LightningBaseModel.on_validation_end "Permalink to this definition")  
Called at the end of validation.

test\_step(*batch*, *batch\_idx*)[¶](#models.pytorch.lightning_models.LightningBaseModel.test_step "Permalink to this definition")  
Operates on a single batch of data from the test set. In this step you’d
normally generate examples or calculate anything of interest such as
accuracy.

Args:  
batch: The output of your `DataLoader`. batch\_idx: The index of this
batch. dataloader\_id: The index of the dataloader that produced this
batch.

> <div>
>
> (only if multiple test dataloaders used).
>
> </div>

Return:  
Any of.

> <div>
>
> -   Any object or value
>
> -   `None` - Testing will skip to the next batch
>
> </div>

<div class="highlight-python notranslate">

<div class="highlight">

    # if you have one test dataloader:
    def test_step(self, batch, batch_idx):
        ...


    # if you have multiple test dataloaders:
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        ...

</div>

</div>

Examples:

<div class="highlight-default notranslate">

<div class="highlight">

    # CASE 1: A single test dataset
    def test_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        out = self(x)
        loss = self.loss(out, y)

        # log 6 example images
        # or generated text... or whatever
        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

</div>

</div>

If you pass in multiple test dataloaders,
[`test_step()`](#models.pytorch.lightning_models.LightningBaseModel.test_step "models.pytorch.lightning_models.LightningBaseModel.test_step")
will have an additional argument. We recommend setting the default value
of 0 so that you can quickly switch between single and multiple
dataloaders.

<div class="highlight-python notranslate">

<div class="highlight">

    # CASE 2: multiple test dataloaders
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx tells you which dataset this is.
        ...

</div>

</div>

Note:  
If you don’t need to test you don’t need to implement this method.

Note:  
When the
[`test_step()`](#models.pytorch.lightning_models.LightningBaseModel.test_step "models.pytorch.lightning_models.LightningBaseModel.test_step")
is called, the model has been put in eval mode and PyTorch gradients
have been disabled. At the end of the test epoch, the model goes back to
training mode and gradients are enabled.

training\_step(*batch*, *batch\_idx*)[¶](#models.pytorch.lightning_models.LightningBaseModel.training_step "Permalink to this definition")  
Here you compute and return the training loss and some additional
metrics for e.g. the progress bar or logger.

Args:  
batch (`Tensor` \| (`Tensor`, …) \| \[`Tensor`, …\]):  
The output of your `DataLoader`. A tensor, tuple or list.

batch\_idx (`int`): Integer displaying index of this batch

Return:  
Any of.

-   `Tensor` - The loss tensor

-   `dict` - A dictionary. Can include any keys, but must include the
    key `'loss'`

-   `None` - Training will skip to the next batch. This is only for automatic optimization.  
    This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

In this step you’d normally do the forward pass and calculate the loss
for a batch. You can also do fancier things like multiple forward passes
or something model specific.

Example:

<div class="highlight-default notranslate">

<div class="highlight">

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        out = self.encoder(x)
        loss = self.loss(out, x)
        return loss

</div>

</div>

To use multiple optimizers, you can switch to ‘manual optimization’ and
control their stepping:

<div class="highlight-python notranslate">

<div class="highlight">

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False


    # Multiple optimizers (e.g.: GANs)
    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()

        # do training_step with encoder
        ...
        opt1.step()
        # do training_step with decoder
        ...
        opt2.step()

</div>

</div>

Note:  
When `accumulate_grad_batches` &gt; 1, the loss returned here will be
automatically normalized by `accumulate_grad_batches` internally.

validation\_step(*batch*, *batch\_idx*)[¶](#models.pytorch.lightning_models.LightningBaseModel.validation_step "Permalink to this definition")  
Operates on a single batch of data from the validation set. In this step
you’d might generate examples or calculate anything of interest like
accuracy.

Args:  
batch: The output of your `DataLoader`. batch\_idx: The index of this
batch. dataloader\_idx: The index of the dataloader that produced this
batch.

> <div>
>
> (only if multiple val dataloaders used)
>
> </div>

Return:  
-   Any object or value

-   `None` - Validation will skip to the next batch

<div class="highlight-python notranslate">

<div class="highlight">

    # if you have one val dataloader:
    def validation_step(self, batch, batch_idx):
        ...


    # if you have multiple val dataloaders:
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ...

</div>

</div>

Examples:

<div class="highlight-default notranslate">

<div class="highlight">

    # CASE 1: A single validation dataset
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        out = self(x)
        loss = self.loss(out, y)

        # log 6 example images
        # or generated text... or whatever
        sample_imgs = x[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

</div>

</div>

If you pass in multiple val dataloaders,
[`validation_step()`](#models.pytorch.lightning_models.LightningBaseModel.validation_step "models.pytorch.lightning_models.LightningBaseModel.validation_step")
will have an additional argument. We recommend setting the default value
of 0 so that you can quickly switch between single and multiple
dataloaders.

<div class="highlight-python notranslate">

<div class="highlight">

    # CASE 2: multiple validation dataloaders
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx tells you which dataset this is.
        ...

</div>

</div>

Note:  
If you don’t need to validate you don’t need to implement this method.

Note:  
When the
[`validation_step()`](#models.pytorch.lightning_models.LightningBaseModel.validation_step "models.pytorch.lightning_models.LightningBaseModel.validation_step")
is called, the model has been put in eval mode and PyTorch gradients
have been disabled. At the end of validation, the model goes back to
training mode and gradients are enabled.

&nbsp;

*class *models.pytorch.lightning\_models.LightningTransformerPredictor(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.lightning_models.LightningTransformerPredictor "Permalink to this definition")  
\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.lightning_models.LightningTransformerPredictor.__init__ "Permalink to this definition")  

forward(*x*)[¶](#models.pytorch.lightning_models.LightningTransformerPredictor.forward "Permalink to this definition")  
Same as `torch.nn.Module.forward()`.

Args:  
[\*](#id5)args: Whatever you decide to pass into the forward method.
[\*\*](#id7)kwargs: Keyword arguments are also possible.

Return:  
Your model’s output

&nbsp;

*class *models.pytorch.lightning\_models.LightningTransformerSequenceClassifier(*\*args: Any*, *\*\*kwargs: Any*)[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier "Permalink to this definition")  
Transformer-based Sequence Classifier

\_\_init\_\_(*\*\*kwargs*)[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier.__init__ "Permalink to this definition")  

forward(*inputs*)[¶](#models.pytorch.lightning_models.LightningTransformerSequenceClassifier.forward "Permalink to this definition")  
Forward pass through the model

</div>

</div>

</div>

<div id="indices-and-tables" class="section">

# Indices and tables[¶](#indices-and-tables "Permalink to this headline")

-   [Index](genindex.html)

-   [Module Index](py-modindex.html)

-   [Search Page](search.html)

</div>

</div>

</div>

</div>

<div class="sphinxsidebar" role="navigation"
aria-label="main navigation">

<div class="sphinxsidebarwrapper">

©2023, Asad Bin Imtiaz, Felix Schlatter. \| Powered by [Sphinx
4.5.0](http://sphinx-doc.org/) & [Alabaster
0.7.12](https://github.com/bitprophet/alabaster)

</div>
