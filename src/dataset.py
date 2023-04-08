import sys

sys.path.insert(0, '../src')
from config import *

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def augment_landmarks(landmarks, max_shift=0.05):
    augmented_landmarks = [[x + np.random.uniform(-max_shift, max_shift), y + np.random.uniform(-max_shift, max_shift)] for x, y in landmarks]
    return augmented_landmarks
    
def frame_dropout(landmarks, dropout_rate=0.1):
    keep_rate = 1 - dropout_rate
    keep_indices = np.random.choice(len(landmarks), int(len(landmarks) * keep_rate), replace=False)
    keep_indices.sort()
    dropped_landmarks = [landmarks[i] for i in keep_indices]
    return dropped_landmarks
    

class ASL_DATSET(Dataset):
    
    # Constructor method
    def __init__(self, transform=None, max_seq_length=MAX_SEQUENCES, augment=False):
        super().__init__()
        
        self.transform = transform
        self.augment   = augment
        
        #[TODO] get this from data
        self.max_seq_length = max_seq_length
        
        self.load_data()
        
    # Load the data method
    def load_data(self):
        
        # Load Processed data
        self.df_train = pd.read_csv(os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,TRAIN_CSV_FILE))
        
        # Generate Absolute path to locate landmark files
        self.file_paths = np.array([os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,x) for x in self.df_train["path"].values])
        
        # Store individual metadata lists
        # [TODO] Cleanup unnecessary files, do we need these?
        self.participant_ids = self.df_train["participant_id"].values
        self.sequence_ids = self.df_train["sequence_id"].values
        self.target = self.df_train['target'].values
        self.size = self.df_train['size'].values
        
    # Get the length of the dataset
    def __len__(self):
        return len(self.df_train)
    
    # Get a single item from the dataset
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Get the processed data for the single index
        landmark_file = self.file_paths[idx]
        
        # Read in the processed file
        landmark_file = torch.load(landmark_file)
        
        # Get the processed landmarks and target for the data
        landmarks = landmark_file
        target = self.target[idx]
        size   = self.size[idx]
        
        # Pad the landmark data
        pad_len = max(0, self.max_seq_length - len(landmarks))
        #landmarks = landmarks + [[[0,0]]*len(landmarks[0])] * pad_len
        padding = np.zeros((pad_len, landmarks.shape[1], landmarks.shape[2]))
        landmarks = np.vstack([landmarks, padding])
        
        if self.transform:
            landmarks = self.transform(landmarks)
            
        if self.augment:
            landmarks = augment_landmarks(landmarks)
            landmarks = frame_dropout(landmarks)
        
        sample = {'landmarks': landmarks, 'target': target, 'size': size}
        
        return sample
    
    # Return a string representation of the dataset
    def __repr__(self):
        return f'ASL_DATSET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'

