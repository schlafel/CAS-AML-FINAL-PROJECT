import sys

sys.path.insert(0, '../src')
from config import *

import os
import json
import pandas as pd
import numpy as np
import mediapipe as mp
import pyarrow.parquet as pq

import torch
import torch.nn.functional as F

from tqdm import tqdm

def preprocess_raw_data(sample=100000):
	
    """
    This method preprocess_data preprocesses the data for easier and faster loading during training time. The data is processed and stored in PROCESSED_DATA_DIR if not already done.
    Parameters:
    max_seq_length: (default=MAX_SEQUENCES) An integer representing the maximum sequence length.
    Functionality:
    If the preprocessed data already exists, the method prints "Preprocessed data found. Skipping...".
    The method first reads the metadata CSV file for training data to obtain a dictionary that maps target values to integer indices. Then, it reads the training data CSV file and generates the absolute path to locate landmark files.
    Next, the method keeps text signs and their respective indices and initializes a list to store the processed data. The data is then processed and stored in the list by iterating over each file path in the training data and reading in the parquet file for that file path. The landmark data is then processed and padded to have a length of max_seq_length. Finally, a dictionary with the processed data is created and added to the list.
    The processed data is saved to disk using the PyTorch torch.save method and the saved file is printed.
    """
    
    if os.path.exists(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, MARKER_FILE)):
        print('Preprocessed data found. Skipping...')
        return
    
    # Check if the landmarks directory exists and create if absent
    if not os.path.exists(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,LANDMARK_FILES)):
        os.makedirs(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,LANDMARK_FILES))
    
    # Read the Metadata CVS file for training data
    label_dict = json.load(open(os.path.join(ROOT_PATH,RAW_DATA_DIR,MAP_JSON_FILE)))

    # Read the Mapping JSON file to map target values to integer indices
    df_train = pd.read_csv(os.path.join(ROOT_PATH,RAW_DATA_DIR,TRAIN_CSV_FILE))[:sample]
    
    # Generate Absolute path to locate landmark parquet files
    file_paths = np.array([os.path.join(ROOT_PATH, RAW_DATA_DIR, x) for x in df_train["path"].values])

    # Generate Absolute path to store landmark processed files
    participant_ids = df_train["participant_id"].values
    sequence_ids = df_train["sequence_id"].values
    processed_files = np.array([str(x)+'-'+str(y)+'.pt' for (x,y) in zip(participant_ids,sequence_ids)])
    
    # keep tect signs and their respective indices
    signs   = df_train["sign"].values
    targets = df_train["sign"].map(label_dict).values

    # Keep track of sequence sizes
    size = []

    # Process the data and return result it
    for i, idx in tqdm(enumerate(range(len(df_train))), total=len(df_train)):
        
        sample = preprocess_data_item(raw_landmark_path=file_paths[idx], targets_sign=targets[idx])    
        
        # Save the processed data to disk
        torch.save(sample['landmarks'], os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,LANDMARK_FILES,processed_files[idx]))
        size.append(sample['size'])
        
    df_train["path"] = [LANDMARK_FILES+'/'+ f for f in processed_files]
    df_train["size"] = size
    df_train["target"] = targets
    
    
    df_train.to_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, TRAIN_CSV_FILE), sep=',',index=False)
    
    # Create the marker file
    with open(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, MARKER_FILE), 'w') as f:
        f.write('')
    
    print(f'Preprocessed data saved in {os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, LANDMARK_FILES)}.')
    

def preprocess_data_item(raw_landmark_path,targets_sign):

    """
    This function is a handy function to process all landmark aequences on a particular location. This will come in handy while testing where individual sequences may be provided
    """
    
    # Get the file path for the single index
    landmark_file = raw_landmark_path

    # Read in the parquet file and process the data
    landmarks = pq.read_table(landmark_file).to_pandas()

    # Read individual landmark data
    # As per dataset description 'he MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values'
    landmarks = landmarks[['frame', 'type', 'landmark_index', 'x', 'y']]

    # We do not need all face mesh landmarks, just the ones for face countours
    # We do not need all pose landmarks, not the ones for face     
    # boolean indexing to filter face landmarks 
    mask = ( ((landmarks['type'] != 'face') & (landmarks['type'] != 'pose')) 
           | ((landmarks['type'] == 'face') & landmarks['landmark_index'].isin(USEFUL_FACE_LANDMARKS))
           | ((landmarks['type'] == 'pose') & landmarks['landmark_index'].isin(USEFUL_POSE_LANDMARKS)))
    
    landmarks = landmarks[mask]
    
    landmarks = landmarks.pivot(index='frame', columns=['type', 'landmark_index'], values=['x', 'y'])
    landmarks.columns = [f"{col[1]}-{col[2]}_{col[0]}" for col in landmarks.columns]
    
    # Interpolate missing values using linear interpolation
    landmarks.interpolate(method='linear', inplace=True, limit=3)
    
    # Fill any remaining missing values with 0
    landmarks.fillna(0, inplace=True)
    landmarks.reset_index(inplace=True)

    columns = list(landmarks.columns)
    new_columns = [columns[(i+1) // 2 + (len(columns) ) // 2 * ((i+1) % 2)] for i in range(1, len(columns))]
    landmarks=landmarks[new_columns].values.tolist()
    data_size=len(landmarks)
    landmarks = np.array([[[frame[i], frame[i+1]] for i in range(0, len(frame), 2)] for frame in landmarks])
    
    # Create a dictionary with the processed data
    return {'landmarks': landmarks, 'target': targets_sign, 'size': data_size}
    	
def calculate_landmark_length_stats():
    
    """
    The function loops through each unique sign in the CSV file, selects rows corresponding to the current sign, and calculates the length of landmarks for each image using PyTorch. It then calculates the minimum, maximum, mean, standard deviation, and median of the landmarks for the current sign, and stores the results in a dictionary.
    """
    
    # Read the CSV file 
    df_train = pd.read_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, TRAIN_CSV_FILE))
    
    # Create an empty dictionary to store average landmarks for each sign type
    avg_landmarks = {}

    # Loop through each unique sign in the CSV file
    for sign in df_train['sign'].unique():
        sign_data = []
        
        # Select rows corresponding to the current sign
        sign_rows = df_train[df_train['sign'] == sign]
        
        # Loop through each row of the current sign type
        for _, row in sign_rows.iterrows():
            
            file_path = os.path.join(ROOT_PATH,PROCESSED_DATA_DIR, row['path'])
            data = torch.load(file_path)
            
            # Add the length of landmarks of the current image to the list of current sign data
            sign_data.append(len(np.array(data['landmarks'])))
            
        # Calculate the minimum, maximum, mean, standard deviation, and median of the landmarks for the current sign
        avg_landmarks[sign] = {
            'min': np.nanmin(sign_data, axis=0),
            'max': np.nanmax(sign_data, axis=0),
            'mean': np.nanmean(sign_data, axis=0),
            'std': np.nanstd(sign_data, axis=0),
            'med': np.nanmedian(sign_data, axis=0)
        }

    # Return the dictionary containing average landmarks for each sign type
    return avg_landmarks
    
def calculate_avg_landmark_positions():
    
    """
    The purpose of this function is to compute the average positions of landmarks for left-hand, right-hand, and face for each sign in the training dataset.
    """
    
    df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))
    avg_landmarks_pos = {}
    
    # Loop over each unique sign in the training dataset
    for sign in df_train['sign'].unique():
        sign_rows = df_train[df_train['sign'] == sign]
        lh_sum, rh_sum, face_sum = 0, 0, 0
        lh_count, rh_count, face_count = 0, 0, 0
        
        # Loop over each row (i.e. frame) for the current sign
        for _, row in sign_rows.iterrows():
            file_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, row['path'])
            data = torch.load(file_path)
            landmarks = np.array(data['landmarks'])
            
            # Extract left-hand, right-hand, and face landmarks
            lh_landmarks = landmarks[:, USED_FACE_FEATURES:USED_FACE_FEATURES+USED_HAND_FEATURES, :]
            rh_landmarks = landmarks[:, USED_FACE_FEATURES+USED_HAND_FEATURES+USED_POSE_FEATURES:, :]
            face_landmarks = landmarks[:, :USED_FACE_FEATURES, :]
            
            # Ignore RuntimeWarnings in the calculation of means
            with np.errstate(divide='ignore', invalid='ignore'):
                # Compute the means of the x and y coordinates for left-hand, right-hand, and face landmarks
                lh_mean = np.nanmean(np.nanmean(lh_landmarks, axis=1), axis=0)
                rh_mean = np.nanmean(np.nanmean(rh_landmarks, axis=1), axis=0)
                face_mean = np.nanmean(np.nanmean(face_landmarks, axis=1), axis=0)
                
                # Add the means to the running totals and counts
                lh_sum += lh_mean
                rh_sum += rh_mean
                face_sum += face_mean
                lh_count += np.count_nonzero(~np.isnan(lh_mean))
                rh_count += np.count_nonzero(~np.isnan(rh_mean))
                face_count += np.count_nonzero(~np.isnan(face_mean))
        
        # Compute the average positions of landmarks for left-hand, right-hand, and face
        avg_lh_landmarks_pos = lh_sum / lh_count if lh_count > 0 else np.array([np.nan, np.nan])
        avg_rh_landmarks_pos = rh_sum / rh_count if rh_count > 0 else np.array([np.nan, np.nan])
        avg_face_landmarks_pos = face_sum / face_count if face_count > 0 else np.array([np.nan, np.nan])
        
        # Store the average positions of landmarks in a dictionary for the current sign
        avg_landmarks_pos[sign] = {'left_hand': avg_lh_landmarks_pos,
                                   'right_hand': avg_rh_landmarks_pos,
                                   'face': avg_face_landmarks_pos}
        
    return avg_landmarks_pos
    
def remove_outlier_or_missing_data(landmark_len_dict):
    
    # The function checks if there are more than SKIP_CONSECUTIVE_ZEROS consecutive frames in which both the X and Y coordinates 
    # are 0 (i.e., [0, 0]). If such consecutive frames are found, the function returns True, otherwise it returns False.
    def has_consecutive_zeros(frames):
        consecutive_count = 0
        max_consecutive = SKIP_CONSECUTIVE_ZEROS
        for frame in frames:
            if frame[0][0] == 0 and frame[0][1] == 0:
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
            data = torch.load(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            missing_file = True
            continue
        
        sign = row['sign']
        median_len  = landmark_len_dict[sign]['med']
        average_len = landmark_len_dict[sign]['mean']
        std_len     = landmark_len_dict[sign]['std']
        landmarks = np.array(data['landmarks'])
        landmarks = np.array([[[frame[i].item(), frame[i+1].item()] for i in range(0, len(frame), 2)] for frame in landmarks])
        landmarks_len = len(np.array(data['landmarks']))
        
        # Extract left-hand, right-hand landmarks
        lh_landmarks = landmarks[:, USED_FACE_FEATURES:USED_FACE_FEATURES+USED_HAND_FEATURES, :]
        rh_landmarks = landmarks[:, USED_FACE_FEATURES+USED_HAND_FEATURES+USED_POSE_FEATURES:, :]            
        
        lh_missings = has_consecutive_zeros(lh_landmarks)
        rh_missings = has_consecutive_zeros(rh_landmarks)
        
        # Check if the length of landmark data is an outlier
        MIN_LEN_THRESHOLD = median_len // 3
        MAX_LEN_THRESHOLD = (average_len + (std_len*2))//1
        
        #print(f"{landmarks_len < MIN_LEN_THRESHOLD} or {landmarks_len > MAX_LEN_THRESHOLD} or {lh_missings} or {rh_missings}")
        if landmarks_len < MIN_LEN_THRESHOLD or landmarks_len > MAX_LEN_THRESHOLD or lh_missings or rh_missings or missing_file:
            
            # Delete the processed file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Mark the row for deletion
            rows_to_drop.append(index)

    # Drop marked rows from the DataFrame
    df_train.drop(rows_to_drop, inplace=True)

    # Save the updated DataFrame to the CSV file
    df_train.to_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE), index=False)


    
    