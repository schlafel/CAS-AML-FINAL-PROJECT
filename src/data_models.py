import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import json
import numpy as np
import pytorch_lightning as pl
import sys

sys.path.insert(0, '../src')
from config import *


class ASL_DATSET(Dataset):
    def __init__(self, transform=None, max_seq_length=MAX_SEQUENCES, ):
        super().__init__()

        self.transform = transform

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

        self.n_features = HAND_FEATURES * 2 + 12 + len(FACE_INDICES)

        self.total_length = self.max_seq_length * self.n_features
        self.load_data()

    def load_data(self):

        # Load Processed data
        self.df_train = pd.read_csv(os.path.join(ROOT_PATH, RAW_DATA_DIR, "train.csv"))
        self.label_dict = json.load(open(os.path.join(ROOT_PATH, RAW_DATA_DIR, MAP_JSON_FILE)))

        # Generate Absolute path to locate landmark files
        self.file_paths = np.array([os.path.join(ROOT_PATH, RAW_DATA_DIR, x) for x in self.df_train["path"].values])
        self.labels = self.df_train.sign.map(self.label_dict).values

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
        landmark_path = self.file_paths[idx]
        target = self.labels[idx]

        # Read in the processed file
        df_in = pd.read_parquet(landmark_path).fillna(0)

        # get number of frames
        n_frames = df_in.frame.nunique()

        # select the landmarks
        landmarks = df_in.loc[(
                                      ((df_in.type == "pose") & (df_in.landmark_index.isin(list(range(11, 23))))) |

                                      ((df_in.type == "face") & (df_in.landmark_index.isin(FACE_INDICES))) |
                                      ((df_in.type == "right_hand")) |
                                      ((df_in.type == "left_hand"))
                              ), COLUMNS_TO_USE
        ].values

        # print(n_frames)
        # pad or crop series to max_seq_length
        if n_frames < self.max_seq_length:
            landmarks = np.append(landmarks, np.zeros(((self.max_seq_length - n_frames) * self.n_features, 2)), axis=0)
        else:
            # crop
            landmarks = landmarks[:self.total_length, :]

        if landmarks.shape[0] != self.total_length:
            print("Wrong length... ", landmark_path, n_frames)

        # landmark_file = torch.load(landmark_file)

        # Get the processed landmarks and target for the data
        # landmarks = landmark_file['landmarks']
        # target = landmark_file['target']
        # size = landmark_file['size']

        # Pad the landmark data
        # pad_len = max(0, self.max_seq_length - len(landmarks))
        # landmarks = landmarks + [[0]*len(landmarks[0])] * pad_len

        # if self.transform:
        #    sample = self.transform(landmarks)

        # create tensor
        lm = torch.from_numpy(landmarks).float().reshape(self.max_seq_length, self.n_features * 2)

        return {'landmarks': lm, 'target': torch.Tensor([target]).long()}

    def __repr__(self):
        return f'ASL_DATSET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}, Number of Features: {self.n_features}, " Number of Frames: {self.max_seq_length}"'


class ASLDataModule(pl.LightningDataModule):
    def __init__(self,
                 max_seq_length = MAX_SEQUENCES,
                 batch_size=16,
                 num_workers=0):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = ASL_DATSET(max_seq_length=self.max_seq_length)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)
        return train_loader


if __name__ == '__main__':

    ds = ASL_DATSET()

    data_loader = DataLoader(ds, batch_size=1,
                             shuffle=True)

    for sample in data_loader:
        pass
        break
