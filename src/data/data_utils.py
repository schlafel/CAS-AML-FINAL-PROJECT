"""
=================================
Data processing Utils description
=================================

This module handles the loading and preprocessing of data. It is specifically tailored for loading ASL sign language
dataset where the raw data includes information about the position of hands, face, and body over time.

ASL stands for American Sign Language, which is a natural language used by individuals who are deaf or hard of hearing
to communicate through hand gestures and facial expressions.

The dataset consists of sequences of frames, where each frame contains multiple "landmarks". Each of these landmarks
has multiple features, such as coordinates. The landmarks may represent various aspects of human body, such as facial
features, hand positions, and body pose.

This module is used to process the raw data, to create a uniform dataset where all sequences are of the same length and
all missing values have been handled in a way that maintains the integrity of the data. This involves steps like
detecting and removing empty frames, selecting specific landmarks, resizing sequences and handling NaN values.
"""

import sys

sys.path.insert(0, '..')
from src.config import *

import os
import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from data.dataset import ASL_DATASET, label_dict_inference, label_dict
from dl_utils import get_dataloader

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

def load_relevant_data_subset(pq_path):
    """
    This function serves a key role in handling data in our pipeline by loading only a subset of the relevant data
    from a given path. The primary purpose of this is to reduce memory overhead when working with large datasets.
    The implementation relies on efficient data loading strategies, leveraging the speed of Parquet file format
    and the ability to read in only necessary chunks of data instead of the whole dataset.

    The function takes as input a string which represents the path to the data file. It makes use of pandas' parquet read function
    to read the data file. This function is particularly suited for reading large datasets as it allows for efficient on-disk storage
    and fast query capabilities. The function uses PyArrow library as the engine for reading the parquet files which ensures efficient
    and fast reading of data. After reading the data, the function selects the relevant subset based on certain criteria, which is
    task specific.

    Args:
        pq_path (str): Path to the data file.

    Returns:
        np.ndarray: Subset of the relevant data as a NumPy array.

    Functionality:
        Loads a subset of the relevant data from a given path.

    :returns: Subset of the relevant data.
    :rtype: np.ndarray

    :param pq_path: Path to the data file.
    :type pq_path: str

    .. note::
    The function assumes that the data file is in parquet format and the necessary libraries for reading
    parquet files are installed. It also assumes that the path provided is a valid path to the data file.
    """
    data_columns = COLUMNS_TO_USE
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def interpolate_missing_values(arr, max_gap=INTEREMOLATE_MISSING):
    """
    This function provides a solution for handling missing values in the data array. It interpolates these missing
    values, filling them with plausible values that maintain the overall data integrity. The function uses a linear
    interpolation method that assumes a straight line between the two points on either side of the gap. The maximum
    gap size for which interpolation should be performed is also configurable.

    Arguments:
    The function takes two arguments - an array with missing values, and a maximum gap size for interpolation. If the
    size of the gap (i.e., number of consecutive missing values) is less than or equal to this specified maximum gap
    size, the function will fill it with interpolated values. This ensures that the data maintains its continuity
    without making too far-fetched estimations for larger gaps.

    Args:
        arr (np.ndarray): Input array with missing values.
        max_gap (int, optional): Maximum gap to fill. Defaults to INTEREMOLATE_MISSING.

    Returns:
        np.ndarray: Array with missing values interpolated.

    Functionality:
        Interpolates missing values in the array. The function fills gaps of up to a maximum size
        with interpolated values, maintaining data integrity and continuity.

    :returns: Array with missing values interpolated.
    :rtype: np.ndarray

    :param arr: Input array with missing values.
    :type arr: np.ndarray

    :param max_gap: Maximum gap to fill.
    :type max_gap: int

    .. note::
    This function uses linear interpolation to fill the missing values. Other forms of interpolation such as polynomial
    or spline may provide better results for specific types of data. It is also worth noting that no imputation method
    can fully recover original data, and as such, results should be interpreted with caution when working with imputed
    data.
    """
    nan_mask = np.isnan(arr)

    for coord_idx in range(arr.shape[2]):
        for lm_idx in range(arr.shape[1]):
            good_indices = np.where(~nan_mask[:, lm_idx, coord_idx])[0]
            if len(good_indices) == 0:
                continue

            curr_idx = good_indices[0]

            for idx in good_indices[1:]:

                last_idx = curr_idx
                curr_idx = idx

                if curr_idx == last_idx + 1:
                    continue

                left_boundary = last_idx
                right_boundary = curr_idx

                if right_boundary - left_boundary <= max_gap:

                    if left_boundary >= 0 and right_boundary < arr.shape[0]:
                        arr[left_boundary + 1:right_boundary, lm_idx, coord_idx] = np.interp(
                            np.arange(left_boundary + 1, right_boundary),
                            [left_boundary, right_boundary],
                            [arr[left_boundary, lm_idx, coord_idx], arr[right_boundary, lm_idx, coord_idx]]
                        )
                    elif left_boundary == 0 and right_boundary < arr.shape[0]:
                        arr[:right_boundary, lm_idx, coord_idx] = arr[right_boundary, lm_idx, coord_idx]
                    elif left_boundary >= 0 and right_boundary == arr.shape[0]:
                        arr[left_boundary:, lm_idx, coord_idx] = arr[left_boundary, lm_idx, coord_idx]

    return arr


def preprocess_raw_data(sample=100000):
    """
    Preprocesses the raw data, saves it as numpy arrays into processed data directory and updates the metadata CSV file.

    This method preprocess_data preprocesses the data for easier and faster loading during training time. The data is
    processed and stored in PROCESSED_DATA_DIR if not already done.

    This function is responsible for preprocessing raw data. The primary functionality involves converting raw data
    into a format more suitable for the machine learning pipeline, namely NumPy arrays. The function operates on a
    sample of data, allowing for efficient processing of large datasets in manageable chunks. Additionally, this
    function also takes care of persisting the preprocessed data for future use and updates the metadata accordingly.


    Args:
    sample (int): Number of samples to preprocess.
    Default is 100000.

    Functionality:
    - The function reads the metadata CSV file for training data to obtain a dictionary that maps target values to
    integer indices.
    - It then reads the training data CSV file and generates the absolute path to locate landmark files.
    - Next, it keeps text signs and their respective indices and initializes a list to store the processed data.
    - The data is then processed and stored in the list by iterating over each file path in the training data and
    reading in the parquet file for that file path.
    - The landmark data is then processed and padded to have a length of max_seq_length.
    - Finally, a dictionary with the processed data is created and added to the list.
    - The processed data is saved to disk using the np.save method and the saved file is printed.

    :param sample: Number of samples to preprocess.
    :type sample: int, optional, default: 100000

    :returns: None

    .. note::
       If the preprocessed data already exists, the function prints "Preprocessed data found. Skipping..." and exits.
    """

    # Marker file to indicate if data is already pre-processed
    marker_file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, MARKER_FILE)

    if os.path.exists(os.path.join(marker_file_path)):
        print('Preprocessed data found. Skipping...')
        return

    # Path to landmark files directory saved as numpy arrays
    landmarks_dir_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, LANDMARK_FILES)

    # Check if the landmarks directory exists and create if absent
    if not os.path.exists(landmarks_dir_path):
        os.makedirs(landmarks_dir_path)

    # Path to raw json dictionary file
    map_json_file_path = os.path.join(ROOT_PATH, RAW_DATA_DIR, MAP_JSON_FILE)

    # Read the Mapping JSON file to map target values to integer indices
    with open(map_json_file_path) as f:
        label_dict = json.load(f)

    # Path to raw training metadata file
    train_csv_file_path = os.path.join(ROOT_PATH, RAW_DATA_DIR, TRAIN_CSV_FILE)

    # Read the Metadata CVS file for training data
    df_train = pd.read_csv(train_csv_file_path)[:sample]

    # Generate Absolute path to locate landmark parquet files
    file_paths = np.array([os.path.join(ROOT_PATH, RAW_DATA_DIR, x) for x in df_train["path"].values])

    # Generate Absolute path to store landmark processed files
    processed_files = np.array(
        [f"{x}-{y}.npy" for (x, y) in zip(df_train["participant_id"].values, df_train["sequence_id"].values)])

    # keep target signs and their respective indices
    signs = df_train["sign"].values
    targets = df_train["sign"].map(label_dict).values

    # Keep track of sequence sizes
    size = []
    orig_size = []
    usable_size = []

    # Create lists for all landmarks
    lm_list = []

    # Process the data and return result it
    for i, idx in tqdm(enumerate(range(len(df_train))), total=len(df_train)):
        sample = preprocess_data_item(raw_landmark_path=file_paths[idx], targets_sign=targets[idx])

        # Save the processed data to disk as numpy arrays
        np.save(os.path.join(landmarks_dir_path, processed_files[idx]), sample['landmarks'])

        lm_list.append(np.where(sample['landmarks']==0,np.nan,sample['landmarks']))

        size.append(sample['size'])
        orig_size.append(sample['orig_size'])
        usable_size.append(sample['usable_size'])

    df_train["path"] = [LANDMARK_FILES + '/' + f for f in processed_files]
    df_train["size"] = size
    df_train["orig_size"] = orig_size
    df_train["target"] = targets
    df_train["usable_size"] = usable_size

    train_csv_output_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE)
    df_train.to_csv(train_csv_output_path, sep=',', index=False)

    # Calculate the statistics
    lm_array = np.concatenate(lm_list)
    lm_means = np.nanmean(lm_array, axis=0)
    lm_std   = np.nanstd (lm_array, axis=0)

    np.save(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "mean.npy"), lm_means)
    np.save(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, "std.npy" ), lm_std)

    # Create the marker file
    with open(marker_file_path, 'w') as f:
        f.write('')

    print(f'Preprocessed data saved in {os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, LANDMARK_FILES)}.')


def preprocess_data_item(raw_landmark_path, targets_sign):
    """
    The function preprocesses landmark data for a single file. The process involves applying transformations to
    raw landmark data to convert it into a form more suitable for machine learning models. The transformations
    may include normalization, scaling, etc. The target sign associated with the landmark data is also taken as input.

    This function is a handy function to process all landmark aequences on a particular location. This will come in
    handy while testing where individual sequences may be provided

    Args:
        raw_landmark_path: Path to the raw landmark file
        targets_sign: The target sign for the given landmark data

    Returns:
    dict: A dictionary containing the preprocessed landmarks, target, and size.

    Functionality:
    - The function reads the parquet file and processes the data.
    - It filters columns to include only frame, type, landmark_index, x, and y.
    - The function then filters face mesh landmarks and pose landmarks based on the predefined useful landmarks.
    - Landmarks data is pivoted to have a multi-level column structure on landmark type and frame sequence ids.
    - Missing values are interpolated using linear interpolation, and any remaining missing values are filled with 0.
    - The function rearranges columns and calculates the number of frames in the data.
    - X and Y coordinates are brought together, and a dictionary with the processed data is created and returned.

    :param raw_landmark_path: Path to the raw landmark file.
    :type raw_landmark_path: str
    :param targets_sign: The target sign for the given landmark data.
    :type targets_sign: int

    :returns: A dictionary containing the preprocessed landmarks, target, and size.
    :rtype: dict
    """

    # Read in the parquet file and process the data
    landmarks = load_relevant_data_subset(raw_landmark_path)

    landmarks, size, orig_size, usable_size = preprocess_data_to_same_size(landmarks)

    # Create a dictionary with the processed data
    return {'landmarks': landmarks, 'target': targets_sign, 'size': size, 'orig_size': orig_size,
            'usable_size': usable_size}


def preprocess_data_to_same_size(landmarks):
    """
    This function preprocesses the input data to ensure all data arrays have the same size, specified by the global
    INPUT_SIZE variable. This uniform size is necessary for subsequent processing and analysis stages, particularly
    those involving machine learning models which often require consistent input sizes. The preprocessing involves
    several steps, including handling missing values, upsampling, and reshaping arrays. It begins by interpolating any
    missing values, and then it subsets the data by selecting only non-empty frames. Various strategies are applied to
    align the data size to the desired INPUT_SIZE, including padding, repeating, and pooling the data.

    Args:
        landmarks (np.ndarray): The input array with landmarks data.

    Returns:
        Tuple[np.ndarray, int, int, int]: A tuple containing processed landmark data, the set input size, the number of
        original frames, and the number of frames after preprocessing.

    :param landmarks: The input array with landmarks data.
    :type landmarks: np.ndarray

    :returns: A tuple containing processed landmark data, the set input size, the number of original frames, and the
    number of frames after preprocessing.
    :rtype: Tuple[np.ndarray, int, int, int]
    """
    num_orig_frames = landmarks.shape[0]

    landmarks[:, USEFUL_HAND_LANDMARKS] = interpolate_missing_values(landmarks[:, USEFUL_HAND_LANDMARKS])

    frames_hands_nansum = np.nanmean(landmarks[:, USEFUL_HAND_LANDMARKS], axis=(1, 2))
    non_empty_frames_idxs = np.where(frames_hands_nansum > 0)[0]
    landmark_data = landmarks[non_empty_frames_idxs]

    landmark_data = landmark_data[:, USEFUL_ALL_LANDMARKS]

    num_frames = landmark_data.shape[0]

    if num_frames == 0:
        landmark_data = np.zeros((INPUT_SIZE, landmark_data.shape[1], landmark_data.shape[2]))

    elif num_frames < INPUT_SIZE:
        new_frame_indices = np.linspace(0, num_frames - 1, INPUT_SIZE)
        upsampled_landmark_data = np.empty((INPUT_SIZE, landmark_data.shape[1], landmark_data.shape[2]))
        for lm_idx in range(landmark_data.shape[1]):
            for coord_idx in range(landmark_data.shape[2]):
                upsampled_landmark_data[:, lm_idx, coord_idx] = np.interp(
                    new_frame_indices, np.arange(num_frames), landmark_data[:, lm_idx, coord_idx]
                )
        landmark_data = upsampled_landmark_data
        size = INPUT_SIZE
    elif num_frames > INPUT_SIZE:
        if num_frames < INPUT_SIZE ** 2:
            repeats = INPUT_SIZE * INPUT_SIZE // num_frames
            landmark_data = np.repeat(landmark_data, repeats=repeats, axis=0)

        pool_size = len(landmark_data) // INPUT_SIZE
        if len(landmark_data) % INPUT_SIZE > 0:
            pool_size += 1

        if pool_size == 1:
            pad_size = (pool_size * INPUT_SIZE) - len(landmark_data)
        else:
            pad_size = (pool_size * INPUT_SIZE) % len(landmark_data)

        pad_left = pad_size // 2 + INPUT_SIZE // 2
        pad_right = pad_size // 2 + INPUT_SIZE // 2
        if pad_size % 2 > 0:
            pad_right += 1

        landmark_data = np.concatenate((np.repeat(landmark_data[:1], repeats=pad_left, axis=0), landmark_data), axis=0)
        landmark_data = np.concatenate((landmark_data, np.repeat(landmark_data[-1:], repeats=pad_right, axis=0)),
                                       axis=0)

        landmark_data = landmark_data.reshape(INPUT_SIZE, -1, N_LANDMARKS, N_DIMS)
        landmark_data = np.nanmean(landmark_data, axis=1)

    size = INPUT_SIZE

    landmark_data = np.where(np.isnan(landmark_data), 0.0, landmark_data)

    return landmark_data, size, num_orig_frames, num_frames


def preprocess_data(landmarks):
    """
    This function preprocesses the input data by applying similar steps as the preprocess_data_to_same_size function,
    but with the difference that it does not interpolate missing values. The function again targets to adjust the size
    of the input data to align with the INPUT_SIZE. It selects only non-empty frames and follows similar strategies of
    padding, repeating, and pooling the data for size alignment.

    Args:
        landmarks (np.ndarray): The input array with landmarks data.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing processed landmark data and the final size of the data.

    :param landmarks: The input array with landmarks data.
    :type landmarks: np.ndarray

    :returns: A tuple containing processed landmark data and the final size of the data.
    :rtype: Tuple[np.ndarray, int]
    """
    frames_hands_nansum = np.nanmean(landmarks[:, USEFUL_HAND_LANDMARKS], axis=(1, 2))
    non_empty_frames_idxs = np.where(frames_hands_nansum > 0)[0]
    landmark_data = landmarks[non_empty_frames_idxs]

    landmark_data = landmark_data[:, USEFUL_ALL_LANDMARKS]

    num_frames = landmark_data.shape[0]

    if num_frames < INPUT_SIZE:
        non_empty_frames_idxs = np.pad(non_empty_frames_idxs, (0, INPUT_SIZE - num_frames), constant_values=-1)
        landmark_data = np.pad(landmark_data, ((0, INPUT_SIZE - num_frames), (0, 0), (0, 0)), constant_values=0)
        size = num_frames
    else:
        if num_frames < INPUT_SIZE ** 2:
            repeats = INPUT_SIZE * INPUT_SIZE // num_frames
            landmark_data = np.repeat(landmark_data, repeats=repeats, axis=0)

        pool_size = len(landmark_data) // INPUT_SIZE
        if len(landmark_data) % INPUT_SIZE > 0:
            pool_size += 1

        if pool_size == 1:
            pad_size = (pool_size * INPUT_SIZE) - len(landmark_data)
        else:
            pad_size = (pool_size * INPUT_SIZE) % len(landmark_data)

        pad_left = pad_size // 2 + INPUT_SIZE // 2
        pad_right = pad_size // 2 + INPUT_SIZE // 2
        if pad_size % 2 > 0:
            pad_right += 1

        landmark_data = np.concatenate((np.repeat(landmark_data[:1], repeats=pad_left, axis=0), landmark_data), axis=0)
        landmark_data = np.concatenate((landmark_data, np.repeat(landmark_data[-1:], repeats=pad_right, axis=0)),
                                       axis=0)

        landmark_data = landmark_data.reshape(INPUT_SIZE, -1, N_LANDMARKS, N_DIMS)
        landmark_data = np.nanmean(landmark_data, axis=1)

        size = INPUT_SIZE

    landmark_data = np.where(np.isnan(landmark_data), 0.0, landmark_data)

    return landmark_data, size


def calculate_landmark_length_stats():
    """
    Calculate statistics of landmark lengths for each sign type.

    Returns:
    dict: A dictionary of landmark lengths for each sign type containing:
    - minimum
    - maximum
    - mean
    - median
    - standard deviation

    Functionality:
    - The function reads the CSV file.
    - It groups the DataFrame by sign.
    - An empty dictionary is created to store average landmarks for each sign type.
    - The function loops through each unique sign and its corresponding rows in the grouped DataFrame.
    - For each sign, it initializes a list to store the length of landmarks for each example of the current sign.
    - It loops through each row of the current sign type, loads the data, and adds the length of landmarks of the
    current example to the list of current sign data.
    - The function calculates the minimum, maximum, mean, standard deviation, and median of the landmarks for the
    current sign and updates the dictionary.
    - The resulting dictionary containing average landmarks for each sign type is returned.

    :returns: A dictionary of landmark lengths for each sign type containing minimum, maximum, mean, median & standard
    deviation
    :rtype: dict
    """

    # Read the CSV file
    df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))

    # Group the dataframe by sign
    grouped_signs = df_train.groupby('sign')

    # Create an empty dictionary to store average landmarks for each sign type
    avg_landmarks = {}

    # Loop through each unique sign and its corresponding rows in the grouped dataframe
    for i, (sign, sign_rows) in tqdm(enumerate(grouped_signs), total=len(grouped_signs)):

        # Initialize a list to store the length of landmarks for each example of the current sign
        sign_data = []

        # Loop through each row of the current sign type
        for _, row in sign_rows.iterrows():
            size = row['size']

            # Add the length of landmarks of the current example to the list of current sign data
            sign_data.append(size)

        # Calculate the minimum, maximum, mean, standard deviation, and median of the landmarks for the current sign
        avg_landmarks[sign] = {
            'min': np.nanmin(sign_data),
            'max': np.nanmax(sign_data),
            'mean': np.nanmean(sign_data),
            'std': np.nanstd(sign_data),
            'med': np.nanmedian(sign_data)
        }

    # Return the dictionary containing average landmarks for each sign type
    return avg_landmarks


def calculate_avg_landmark_positions(dataset):
    """
    Calculate the average landmark positions for left-hand, right-hand, and face landmarks for each sign in the dataset.
    The purpose of this function is to compute the average positions of landmarks for left-hand, right-hand, and face
    for each sign in the training dataset.

    Returns:
    List : Containing a dictionary with average x/y positions with keys
    -   'left_hand'
    -   'right_hand'
    -   'face'

    Functionality:
    - The function takes an ASLDataset object as an input, which contains the training data.
    - It calculates the average landmark positions for left-hand, right-hand, and face landmarks for each sign in the
    dataset.
    - The function returns a list containing a dictionary with average x/y positions with keys 'left_hand',
    'right_hand', and 'face' for each sign.

    :param dataset: The ASL dataset object containing the training data.
    :type dataset: ASL_DATASET

    :return: A list containing a dictionary with average x/y positions with keys 'left_hand', 'right_hand', and 'face'
    for each sign.
    :rtype: List[Dict[str, np.ndarray]]
    """
    df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))
    avg_landmarks_pos = {}

    # Loop over each unique sign in the training dataset
    signs = df_train['sign'].unique()
    for c, sign in tqdm(enumerate(signs), total=len(signs)):
        sign_rows = df_train[df_train['sign'] == sign]
        lh_sum, rh_sum, face_sum = np.zeros(2), np.zeros(2), np.zeros(2)
        lh_count, rh_count, face_count = 0, 0, 0

        # Loop over each row (i.e., frame) for the current sign
        for _, row in sign_rows.iterrows():
            file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, row['path'])
            data = np.load(file_path)
            landmarks = data

            # Extract left-hand, right-hand, and face landmarks
            lh_landmarks = landmarks[:, LEFT_HAND_INDICES, :]
            rh_landmarks = landmarks[:, RIGHT_HAND_INDICES, :]
            face_landmarks = landmarks[:, FACE_INDICES, :]

            # Compute the means of the x and y coordinates for left-hand, right-hand, and face landmarks
            lh_mean = np.nanmean(lh_landmarks, axis=(0, 1))
            rh_mean = np.nanmean(rh_landmarks, axis=(0, 1))
            face_mean = np.nanmean(face_landmarks, axis=(0, 1))

            # Add the means to the running totals and increment counts
            lh_sum += lh_mean
            rh_sum += rh_mean
            face_sum += face_mean
            lh_count += 1
            rh_count += 1
            face_count += 1

        # Compute the average positions of landmarks for left-hand, right-hand, and face
        avg_lh_landmarks_pos = lh_sum / lh_count
        avg_rh_landmarks_pos = rh_sum / rh_count
        avg_face_landmarks_pos = face_sum / face_count

        # Store the average positions of landmarks in a dictionary for the current sign
        avg_landmarks_pos[sign] = {'left_hand' : avg_lh_landmarks_pos,
                                   'right_hand': avg_rh_landmarks_pos,
                                   'face'      : avg_face_landmarks_pos}

    return avg_landmarks_pos


def remove_unusable_data():
    """
    This function checks the existing training data for unusable instances, like missing files or data that is smaller than the set
    minimum sequence length. If unusable data is found, it is removed from the system, both in terms of files and entries in the training
    dataframe. The dataframe is updated and saved back to the disk. If a cleansing marker file exists, it skips the process, indicating
    that the data is already cleaned.

    Functionality:
        The function iterates through the DataFrame rows, attempting to load and check each landmark file specified in the row's path.
        If the file is missing or if the file's usable size is less than a predefined threshold, the function deletes the corresponding
        landmark file and marks the row for deletion in the DataFrame. At the end, the function removes all marked rows from the DataFrame,
        updates it and saves it to the disk.

    Returns:
         None

    :return: None, the function doesn't return anything. It modifies data in-place.
    """
    marker_file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, CLEANED_FILE)

    if os.path.exists(os.path.join(marker_file_path)):
        print('Cleansed data found. Skipping...')
        return

    # Load the training data
    df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))

    # List of row indices to drop
    rows_to_drop = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):



        # Load the file and get the length of its landmarks data
        file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, row['path'])

        missing_file = False
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            missing_file = True
            continue

        usable_size = row['usable_size']

        if (
                usable_size < MIN_SEQUEENCES or missing_file  # Has land mark file missing
        ):

            # Delete the processed file
            if os.path.exists(file_path):
                # print(f"removing {file_path}: landmarks_len {landmarks_len} {landmarks_len < MIN_LEN_THRESHOLD} {landmarks_len > MAX_LEN_THRESHOLD} lh_missings {lh_missings} rh_missings {rh_missings}")
                os.remove(file_path)

            # Mark the row for deletion
            rows_to_drop.append(index)

    # Drop marked rows from the DataFrame
    df_train.drop(rows_to_drop, inplace=True)

    # Save the updated DataFrame to the CSV file
    df_train.to_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE), index=False)

    # Create the marker file
    with open(marker_file_path, 'w') as f:
        f.write('')


def remove_outlier_or_missing_data(landmark_len_dict):
    """
    This function removes rows from the training data that contain missing or outlier landmark data. It takes as input a dictionary
    containing the statistics of landmark lengths for each sign type. The function processes the training data and removes rows with
    missing or outlier landmark data. The function also includes a nested function 'has_consecutive_zeros' which checks for consecutive
    frames where X and Y coordinates are both zero. If a cleansing marker file exists, it skips the process, indicating that the data
    is already cleaned.

    Functionality:
        This function takes a dictionary with the statistics of landmark lengths per sign type and uses it to identify outlier sequences.
        It removes any rows with missing or outlier landmark data. An outlier sequence is defined as one that is either less than a third
        of the median length or more than two standard deviations away from the mean length. A row is also marked for deletion if the
        corresponding landmark file is missing or if the sign's left-hand or right-hand landmarks contain more than a specified number
        of consecutive zeros.

    Args:
        landmark_len_dict (dict): A dictionary containing the statistics of landmark lengths for each sign type.

    Returns:
        None

    :param landmark_len_dict: A dictionary containing the statistics of landmark lengths for each sign type.
    :type landmark_len_dict: dict

    :return: None, the function doesn't return anything. It modifies data in-place.
    """

    marker_file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, CLEANED_FILE)

    if os.path.exists(os.path.join(marker_file_path)):
        print('Cleansed data found. Skipping...')
        return

    # The function checks if there are more than SKIP_CONSECUTIVE_ZEROS consecutive frames in which both the X and Y coordinates
    # are 0 (i.e., [0, 0]). If such consecutive frames are found, the function returns True, otherwise it returns False.

    def has_consecutive_zeros(frames):
        """
        Check if there are consecutive frames with both X and Y coordinates equal to zero.

        Args:
        frames (np.array): Array of landmarks for a given sample.

        Returns: bool

        :param frames: Array of landmarks for a given sample.
        :type frames: np.array

        :return: True if there are consecutive frames with both X and Y coordinates equal to zero, False otherwise.
        :rtype: bool
        """

        consecutive_count = 0
        max_consecutive = SKIP_CONSECUTIVE_ZEROS

        for frame in frames:
            if frame[0][0] == 0. and frame[0][1] == 0.:
                consecutive_count += 1
            else:
                consecutive_count = 0

        if consecutive_count >= max_consecutive and consecutive_count < frames.shape[0]:
            return True

        return False

    # Load the training data
    df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))

    # List of row indices to drop
    rows_to_drop = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):

        missing_file = False

        # Load the file and get the length of its landmarks data
        file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, row['path'])

        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            missing_file = True
            continue

        sign = row['sign']

        median_len = landmark_len_dict[sign]['med']
        average_len = landmark_len_dict[sign]['mean']
        std_len = landmark_len_dict[sign]['std']

        landmarks = np.array(data)
        landmarks_len = len(landmarks)

        # Extract left-hand, right-hand landmarks
        lh_landmarks = landmarks[:, LEFT_HAND_INDICES, :]
        rh_landmarks = landmarks[:, RIGHT_HAND_INDICES, :]

        lh_missings = has_consecutive_zeros(lh_landmarks)
        rh_missings = has_consecutive_zeros(rh_landmarks)

        # Check if the length of landmark data is an outlier
        MIN_LEN_THRESHOLD = median_len // 3
        MAX_LEN_THRESHOLD = (average_len + (std_len * 2)) // 1

        # print(f"{landmarks_len < MIN_LEN_THRESHOLD} or {landmarks_len > MAX_LEN_THRESHOLD} or {lh_missings} or {rh_missings}")
        if (
                landmarks_len < MIN_LEN_THRESHOLD  # Sequences of landmark file are outlier, 3rd of median length
                or landmarks_len > MAX_LEN_THRESHOLD  # Sequences of landmark file are outlier, 2 std away from average length
                or missing_file  # Has land mark file missing
        ):

            # Delete the processed file
            if os.path.exists(file_path):
                # print(f"removing {file_path}: landmarks_len {landmarks_len} {landmarks_len < MIN_LEN_THRESHOLD} {landmarks_len > MAX_LEN_THRESHOLD} lh_missings {lh_missings} rh_missings {rh_missings}")
                os.remove(file_path)

            # Mark the row for deletion
            rows_to_drop.append(index)

    # Drop marked rows from the DataFrame
    df_train.drop(rows_to_drop, inplace=True)

    # Save the updated DataFrame to the CSV file
    df_train.to_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE), index=False)

    # Create the marker file
    with open(marker_file_path, 'w') as f:
        f.write('')


def create_data_loaders(asl_dataset, train_size=TRAIN_SIZE, valid_size=VALID_SIZE, test_size=TEST_SIZE,
                        batch_size=BATCH_SIZE, random_state=SEED, dl_framework = DL_FRAMEWORK,
                        num_workers = os.cpu_count()):
    """
    Split the ASL dataset into training, validation, and testing sets and create data loaders for each set.

    Args:
    asl_dataset (ASLDataset): The ASL dataset to load data from.
    train_size (float, optional): The proportion of the dataset to include in the training set. Defaults to 0.8.
    valid_size (float, optional): The proportion of the dataset to include in the validation set. Defaults to 0.1.
    test_size (float, optional): The proportion of the dataset to include in the testing set. Defaults to 0.1.
    batch_size (int, optional): The number of samples per batch to load. Defaults to BATCH_SIZE.
    random_state (int, optional): The seed used by the random number generator for shuffling the data. Defaults to SEED.

    Returns:
    tuple of DataLoader: A tuple containing the data loaders for training, validation, and testing sets.

    :param asl_dataset: The ASL dataset to load data from.
    :type asl_dataset: ASLDataset
    :param train_size: The proportion of the dataset to include in the training set.
    :type train_size: float
    :param valid_size: The proportion of the dataset to include in the validation set.
    :type valid_size: float
    :param test_size: The proportion of the dataset to include in the testing set.
    :type test_size: float
    :param batch_size: The number of samples per batch to load.
    :type batch_size: int
    :param random_state: The seed used by the random number generator for shuffling the data.
    :type random_state: int
    :return: A tuple containing the data loaders for training, validation, and testing sets.
    :rtype: tuple of DataLoader
    """

    # Split the data into train and test sets
    train_df, test_df = train_test_split(asl_dataset.df_train, test_size=test_size, random_state=random_state,
                                         stratify=asl_dataset.df_train['target'])

    # Split the train set further into train and validation sets
    train_df, valid_df = train_test_split(train_df, test_size=valid_size / (train_size + valid_size),
                                          random_state=random_state, stratify=train_df['target'])

    # Create dataset instances for each split
    train_dataset = ASL_DATASET(metadata_df=train_df)
    valid_dataset = ASL_DATASET(metadata_df=valid_df)
    test_dataset  = ASL_DATASET(metadata_df=test_df)

    # Create data loaders for each split
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True ,dl_framework = dl_framework,num_workers = num_workers)
    valid_loader = get_dataloader(valid_dataset, batch_size=batch_size, shuffle=False,dl_framework = dl_framework,num_workers = num_workers)
    test_loader  = get_dataloader(test_dataset , batch_size=batch_size, shuffle=False,dl_framework = dl_framework,num_workers = num_workers)

    return train_loader, valid_loader, test_loader



if __name__ == '__main__':

    preprocess_raw_data(sample=100000)
    remove_unusable_data()
