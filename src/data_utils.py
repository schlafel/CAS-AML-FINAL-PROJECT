import sys

sys.path.insert(0, '../src')
from config import *

import os
import json
import pandas as pd
import numpy as np
import mediapipe as mp
import pyarrow.parquet as pq

from tqdm import tqdm

def preprocess_raw_data(sample=100000):
	
    """
    Preprocesses the raw data, saves it as numpy arrays into processed data directory and updates the metadata CSV file.

    This method preprocess_data preprocesses the data for easier and faster loading during training time. The data is processed and stored in PROCESSED_DATA_DIR if not already done.
    
    Parameters:
    max_seq_length: (default=MAX_SEQUENCES) An integer representing the maximum sequence length.
    
    Functionality:
    If the preprocessed data already exists, the method prints "Preprocessed data found. Skipping...".
    The method first reads the metadata CSV file for training data to obtain a dictionary that maps target values to integer indices. Then, it reads the training data CSV file and generates the absolute path to locate landmark files.
    Next, the method keeps text signs and their respective indices and initializes a list to store the processed data. The data is then processed and stored in the list by iterating over each file path in the training data and reading in the parquet file for that file path. The landmark data is then processed and padded to have a length of max_seq_length. Finally, a dictionary with the processed data is created and added to the list.
    The processed data is saved to disk using the np.save method and the saved file is printed.

    Args:
    sample (int): Number of samples to preprocess. 
    Default is 100000.
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
    
    # keep tect signs and their respective indices
    signs   = df_train["sign"].values
    targets = df_train["sign"].map(label_dict).values
    
    # Keep track of sequence sizes
    size = []

    # Process the data and return result it
    for i, idx in tqdm(enumerate(range(len(df_train))), total=len(df_train)):
        
        sample = preprocess_data_item(raw_landmark_path=file_paths[idx], targets_sign=targets[idx])    
        
        # Save the processed data to disk as numpy arrays
        np.save(os.path.join(landmarks_dir_path, processed_files[idx]), sample['landmarks'])
        size.append(sample['size'])
        
    df_train["path"] = [LANDMARK_FILES+'/'+ f for f in processed_files]
    df_train["size"] = size
    df_train["target"] = targets
    
    train_csv_output_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE)
    df_train.to_csv(train_csv_output_path, sep=',',index=False)
    
    # Create the marker file
    with open(marker_file_path, 'w') as f:
        f.write('')
    
    print(f'Preprocessed data saved in {os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, LANDMARK_FILES)}.')    
    
def preprocess_data_item(raw_landmark_path,targets_sign):
    """
    Preprocesses the landmark data for a single file. This method is used in pre processing of all data.
    At inference, this method may be called to preprocess the data items in the same manner
    This function is a handy function to process all landmark aequences on a particular location. This will come in handy while testing where individual sequences may be provided
    
    Args:
    raw_landmark_path (str): Path to the raw landmark file.
    targets_sign (int): The target sign for the given landmark data.

    Returns:
    dict: A dictionary containing the preprocessed landmarks, target, and size.
    """
        
    # Read in the parquet file and process the data
    landmarks = pq.read_table(raw_landmark_path).to_pandas()

    # Read individual landmark data
    # As per dataset description the MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values'
    # Filter columns to include only frame, type, landmark_index, x, and y
    landmarks = landmarks[['frame', 'type', 'landmark_index', 'x', 'y']]

    # We do not need all face mesh landmarks, just the ones for face countours
    # We do not need all pose landmarks, not the ones for face     
    # boolean indexing to filter face landmarks 
    mask = ( ((landmarks['type'] != 'face') & (landmarks['type'] != 'pose')) 
           | ((landmarks['type'] == 'face') & landmarks['landmark_index'].isin(USEFUL_FACE_LANDMARKS))
           | ((landmarks['type'] == 'pose') & landmarks['landmark_index'].isin(USEFUL_POSE_LANDMARKS)))
    
    landmarks = landmarks[mask]
    
    # Pivot the dataframe to have a multi-level column structure on landmark type and frame sequence ids
    landmarks = landmarks.pivot(index='frame', columns=['type', 'landmark_index'], values=['x', 'y'])
    landmarks.columns = [f"{col[1]}-{col[2]}_{col[0]}" for col in landmarks.columns]
    
    # Interpolate missing values using linear interpolation
    landmarks.interpolate(method='linear', inplace=True, limit=3)
    
    # Fill any remaining missing values with 0
    landmarks.fillna(0, inplace=True)
    landmarks.reset_index(inplace=True)

    # Rearrange columns
    columns = list(landmarks.columns)
    new_columns = [columns[(i+1) // 2 + (len(columns) ) // 2 * ((i+1) % 2)] for i in range(1, len(columns))]
    landmarks=landmarks[new_columns].values.tolist()
    
    # Calculate the number of frames in the data
    data_size = len(landmarks)
    
    # Bring X and Y coordinates together
    landmarks = np.array([[[frame[i], frame[i+1]] for i in range(0, len(frame), 2)] for frame in landmarks])
    
    # Create a dictionary with the processed data
    return {'landmarks': landmarks, 'target': targets_sign, 'size': data_size}
    	
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
    """
        
    # Read the CSV file 
    df_train = pd.read_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, TRAIN_CSV_FILE))
    
    # Group the dataframe by sign
    grouped_signs = df_train.groupby('sign')
    
    # Create an empty dictionary to store average landmarks for each sign type
    avg_landmarks = {}

    # Loop through each unique sign and its corresponding rows in the grouped dataframe
    for sign, sign_rows in grouped_signs:
        
        # Initialize a list to store the length of landmarks for each example of the current sign
        sign_data = []
        
        # Loop through each row of the current sign type
        for _, row in sign_rows.iterrows():

            file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, row['path'])
            data = np.load(file_path)

            # Add the length of landmarks of the current example to the list of current sign data
            sign_data.append(len(data))

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
    
def calculate_avg_landmark_positions():
    """
    Calculate the average landmark positions for left-hand, right-hand, and face landmarks for each sign in the dataset.
    The purpose of this function is to compute the average positions of landmarks for left-hand, right-hand, and face for each sign in the training dataset.
    
    Returns:
    List : Containing a dictionary with average x/y positions with keys
        'left_hand'
        'right_hand'
        'face'
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
            lh_landmarks = landmarks[:, USED_FACE_FEATURES:USED_FACE_FEATURES + USED_HAND_FEATURES, :]
            rh_landmarks = landmarks[:, USED_FACE_FEATURES + USED_HAND_FEATURES + USED_POSE_FEATURES:, :]
            face_landmarks = landmarks[:, :USED_FACE_FEATURES, :]

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
        avg_landmarks_pos[sign] = {'left_hand': avg_lh_landmarks_pos,
                                   'right_hand': avg_rh_landmarks_pos,
                                   'face': avg_face_landmarks_pos}

    return avg_landmarks_pos
    
def remove_outlier_or_missing_data(landmark_len_dict):
    """
    Remove rows from the training data with missing or outlier landmark data.

    Args:
    landmark_len_dict (dict): A dictionary containing the statistics of landmark lengths for each sign type.

    Returns:
    None
    """
    
    # The function checks if there are more than SKIP_CONSECUTIVE_ZEROS consecutive frames in which both the X and Y coordinates 
    # are 0 (i.e., [0, 0]). If such consecutive frames are found, the function returns True, otherwise it returns False.
    
    def has_consecutive_zeros(frames):
        """
        Check if there are consecutive frames with both X and Y coordinates equal to zero.

        Args:
        frames (np.array): Array of landmarks for a given sample.

        Returns:
        bool: 
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
    for index, row in df_train.iterrows():
        
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
        
        median_len  = landmark_len_dict[sign]['med']
        average_len = landmark_len_dict[sign]['mean']
        std_len     = landmark_len_dict[sign]['std']
        
        landmarks = np.array(data)
        landmarks_len = len(landmarks)
        
        # Extract left-hand, right-hand landmarks
        lh_landmarks = landmarks[:, USED_FACE_FEATURES:USED_FACE_FEATURES+USED_HAND_FEATURES, :]
        rh_landmarks = landmarks[:, USED_FACE_FEATURES+USED_HAND_FEATURES+USED_POSE_FEATURES:, :]            
        
        lh_missings = has_consecutive_zeros(lh_landmarks)
        rh_missings = has_consecutive_zeros(rh_landmarks)
        
        # Check if the length of landmark data is an outlier
        MIN_LEN_THRESHOLD = median_len // 3
        MAX_LEN_THRESHOLD = (average_len + (std_len*2))//1
        
        #print(f"{landmarks_len < MIN_LEN_THRESHOLD} or {landmarks_len > MAX_LEN_THRESHOLD} or {lh_missings} or {rh_missings}")
        if (
               landmarks_len < MIN_SEQUENCES      # Sequences of landmark file are too short
            or landmarks_len > MAX_SEQUENCES      # Sequences of landmark file are too long
            or landmarks_len < MIN_LEN_THRESHOLD  # Sequences of landmark file are outlier, 3rd of median length
            or landmarks_len > MAX_LEN_THRESHOLD  # Sequences of landmark file are outlier, 2 std away from average length
            or lh_missings                        # Has 4 or more hand larmarks missing for left hand
            or rh_missings                        # Has 4 or more hand larmarks missing for right hand
            or missing_file                       # Has land mark file missing
        ):
            
            # Delete the processed file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Mark the row for deletion
            rows_to_drop.append(index)

    # Drop marked rows from the DataFrame
    df_train.drop(rows_to_drop, inplace=True)

    # Save the updated DataFrame to the CSV file
    df_train.to_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE), index=False)


if __name__ == '__main__':

    preprocess_raw_data(sample=100000)
    
    landmark_len_dict = calculate_landmark_length_stats()
    max_seq_len = 0
    for label, stats in landmark_len_dict.items():
        seq_len = stats['max']
        if seq_len > max_seq_len:
            max_seq_len = seq_len
    print(max_seq_len)
    
    remove_outlier_or_missing_data(landmark_len_dict)
    
    cleansed_landmark_len_dict = calculate_landmark_length_stats()
    max_seq_len = 0
    for label, stats in cleansed_landmark_len_dict.items():
        seq_len = stats['max']
        if seq_len > max_seq_len:
            max_seq_len = seq_len
    print(max_seq_len)
    
    