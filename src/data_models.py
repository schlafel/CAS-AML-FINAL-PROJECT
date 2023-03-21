import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import json
import numpy as np

import sys
sys.path.insert(0, '../src')
from config import *


class ASL_DATSET(Dataset):
    def __init__(self, transform=None, max_seq_length=MAX_SEQUENCES):
        super().__init__()

        self.transform = transform

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

        self.load_data()

    def load_data(self):

        # Load Processed data
        self.df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))

        # Generate Absolute path to locate landmark files
        self.file_paths = np.array(
            [os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, x) for x in self.df_train["path"].values])

        # Store individual metadata lists
        # [TODO] Cleanup unnecessary files, do we need these?
        self.participant_ids = self.df_train["participant_id"].values
        self.sequence_ids = self.df_train["sequence_id"].values

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()

        # Get the processed data for the single index
        landmark_file = self.file_paths[idx]

        # Read in the processed file
        landmark_file = torch.load(landmark_file)

        # Get the processed landmarks and target for the data
        landmarks = landmark_file['landmarks']
        target = landmark_file['target']
        size = landmark_file['size']


        ## now select only relevant features
        #structure of landmarks

        # Pad the landmark data
        pad_len = max(0, self.max_seq_length - len(landmarks))
        landmarks = landmarks + [[0] * len(landmarks[0])] * pad_len

        if self.transform:
            sample = self.transform(landmarks)

        sample = {'landmarks': landmarks, 'target': target, 'size': size}

        return sample

    def __repr__(self):
        return f'ASL_DATSET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'


if __name__ == '__main__':

    ds = ASL_DATSET()

    data_loader = DataLoader(ds,batch_size = 1,
                             shuffle = True)

    for sample in data_loader:
        pass
        break