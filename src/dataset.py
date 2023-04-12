import sys

sys.path.insert(0, '../src')
from config import *
from augmentations import *

import os
import pandas as pd
import numpy as np
import random
import json

# Label dictionaries
label_dict = json.load(open(os.path.join(ROOT_PATH,RAW_DATA_DIR,MAP_JSON_FILE)))
label_dict_inference = dict(zip(label_dict.values(),label_dict.keys()))
class ASL_DATASET:
    """
    A dataset class for the ASL dataset.
    """

    # Constructor method
    def __init__(self, transform=None, max_seq_length=MAX_SEQUENCES, augment=False):
        """
        Initialize the dataset.

        Args:
            transform (callable, optional): A function/transform to apply to the data.
            max_seq_length (int): The maximum sequence length for the data.
            augment (bool): Whether to apply data augmentation.
        """

        super().__init__()

        self.transform = transform
        self.augment = augment

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

        self.load_data()

    # Load the data method
    def load_data(self):
        """
        Load the data for the dataset.
        """
        # Load Processed data
        self.df_train = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))

        # Generate Absolute path to locate landmark files
        self.file_paths = np.array(
            [os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, x) for x in self.df_train["path"].values])

        # Store individual metadata lists
        self.participant_ids = self.df_train["participant_id"].values
        self.sequence_ids = self.df_train["sequence_id"].values
        self.target = self.df_train['target'].values
        self.size = self.df_train['size'].values

    # Get the length of the dataset
    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.df_train)

    # Get a single item from the dataset
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the landmarks, target, and size for the item.
        """
        #if torch.is_tensor(idx):
        #    idx = idx.item()

        # Get the processed data for the single index
        landmark_file = self.file_paths[idx]

        # Read in the processed file
        landmarks = np.load(landmark_file)

        # Get the processed landmarks and target for the data
        target = self.target[idx]
        size = self.size[idx]

        if self.transform:
            landmarks = self.transform(landmarks)

        if self.augment:
            if random.random() < 0.1:
                landmarks = shift_landmarks(landmarks)
            if random.random() < 0.1:
                landmarks = mirror_landmarks(landmarks)  # TODO : Change Names
            if random.random() < 0.1:
                landmarks = random_scaling(landmarks)
            if random.random() < 0.1:
                landmarks = random_rotation(landmarks)  # TODO : Change Names
            if random.random() < 0.1:
                landmarks = frame_dropout(landmarks)
                size = len(landmarks)

        # Pad the landmark data
        pad_len = max(0, self.max_seq_length - len(landmarks))

        padding = np.zeros((pad_len, landmarks.shape[1], landmarks.shape[2]))
        landmarks = np.vstack([landmarks, padding])

        sample = {'landmarks': landmarks, 'target': target, 'size': size}

        return sample

    # Return a string representation of the dataset
    def __repr__(self):
        """
        Return a string representation of the dataset.
        """

        return f'ASL_DATASET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'


if __name__ == '__main__':

    dataset = ASL_DATASET(augment=True)

    from torch.utils.data import DataLoader
    ## ASL Data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f'Length data loader: {len(dataloader)}')

    sample = next(iter(dataloader))
    frames = sample['landmarks']
    target = sample['target']
    size = sample['size']

    print(f'Sample size: {size}')
    print(f'Sample target: {target}')

    index = 0
    print(f'Total frames: {frames[index].shape}')

    landmark_lists = frames[index][:size[index]]
    print(f'Actual frames: {landmark_lists.shape}')
