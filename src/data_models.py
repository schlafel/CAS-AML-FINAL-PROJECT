import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np


from config import *


class ASL_DATSET(Dataset):
    def __init__(self, transform=None, max_seq_length=MAX_SEQUENCES):
        super().__init__()

        self.transform = transform
        self.load_datas()

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

    def load_datas(self):

        # Read the Metadata CVS file for training data
        self.label_dict = json.load(open(os.path.join(ROOT_PATH, DATA_DIR, MAP_JSON_FILE)))

        # Read the Mapping JSON file to map target values to integer indices
        self.df_train = pd.read_csv(os.path.join(ROOT_PATH, DATA_DIR, TRAIN_CSV_FILE))

        # Generate Absolute path to locate landmark files
        self.file_paths = np.array([os.path.join(ROOT_PATH, DATA_DIR, x) for x in self.df_train["path"].values])

        # Store individual metadata lists
        # [TODO] Cleanup unnecessary files, do we need these?
        self.participant_ids = self.df_train["participant_id"].values
        self.sequence_ids = self.df_train["sequence_id"].values

        # keep tect signs and their respective indices
        self.signs = self.df_train["sign"].values
        self.targets = self.df_train["sign"].map(self.label_dict).values

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()

        # Get the file path for the single index
        landmark_file = self.file_paths[idx]

        # Read in the parquet file and process the data
        landmarks = pq.read_table(landmark_file).to_pandas()

        # Read individual landmark data
        # As per dataset description 'he MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values'
        landmarks = landmarks[['frame', 'type', 'landmark_index', 'x', 'y']]

        landmarks = landmarks.pivot(index='frame', columns=['type', 'landmark_index'], values=['x', 'y'])
        landmarks.columns = [f"{col[1]}-{col[2]}_{col[0]}" for col in landmarks.columns]
        landmarks.reset_index(inplace=True)

        columns = list(landmarks.columns)
        new_columns = [columns[(i + 1) // 2 + (len(columns)) // 2 * ((i + 1) % 2)] for i in range(1, len(columns))]
        landmarks = landmarks[new_columns].values.tolist()

        # Pad the landmark data
        pad_len = max(0, self.max_seq_length - len(landmarks))
        landmarks = landmarks + [[0] * len(new_columns)] * pad_len

        if self.transform:
            sample = self.transform(landmarks)

        sign = self.signs[idx]
        target = self.targets[idx]

        sample = {'landmarks': landmarks, 'target': target}

        return sample

    def __repr__(self):
        return f'ASL_DATSET(Participants: {len(set(self.participant_ids))}, Unique signs: {len(self.label_dict)}), Length: {len(self.df_train)}'
