import sys

sys.path.insert(0, '../src')
from config import *

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Data Augmentation Methods
"""
def shift_landmarks(frames, max_shift=0.01):
    """
    Shift landmark coordinates randomly by a small amount.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        max_shift (float): Maximum shift for the random shift (default: 0.01).

    Returns:
        numpy.ndarray: An array of augmented landmarks.
    """
    h = np.random.uniform(-max_shift, max_shift)
    v = np.random.uniform(-max_shift, max_shift)
    augmented_landmarks = np.array([[[x + h, y + v] for x, y in landmarks] for landmarks in frames])
    return augmented_landmarks


def mirror_landmarks(frames):
    """
    Invert/mirror landmark coordinates along the x-axis.

    Args:
        frames (numpy.ndarray): An array of landmarks data.

    Returns:
        numpy.ndarray: An array of inverted landmarks.
    """
    inverted_frames = frames.copy()
    inverted_frames[..., 0] = np.array(
        [[[((x - 0.5) * (-1)) + 0.5, y] for x, y in landmarks] for landmarks in inverted_frames])
    return inverted_frames


def frame_dropout(frames, dropout_rate=0.05):
    """
    Randomly drop frames from the input landmark data.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        dropout_rate (float): The proportion of frames to drop (default: 0.05).

    Returns:
        numpy.ndarray: An array of landmarks with dropped frames.
    """
    keep_rate = 1 - dropout_rate
    keep_indices = np.random.choice(len(frames), int(len(frames) * keep_rate), replace=False)
    keep_indices.sort()
    dropped_landmarks = np.array([frames[i] for i in keep_indices])
    return dropped_landmarks


def random_scaling(frames, scale_range=(0.9, 1.1)):
    """
    Apply random scaling to landmark coordinates.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        scale_range (tuple): A tuple containing the minimum and maximum scaling factors (default: (0.9, 1.1)).

    Returns:
        numpy.ndarray: An array of landmarks with randomly scaled coordinates.
    """
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return frames * scale_factor


def random_rotation(frames, max_angle=10):
    """
    Apply random rotation to landmark coordinates.

    Args:
        frames (numpy.ndarray): An array of landmarks data.
        max_angle (int): The maximum rotation angle in degrees (default: 10).

    Returns:
        numpy.ndarray: An array of landmarks with randomly rotated coordinates.
    """
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    return np.einsum('ijk,kl->ijl', frames, rotation_matrix)


class ASL_DATSET(Dataset):
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

        if torch.is_tensor(idx):
            idx = idx.item()

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

        return f'ASL_DATSET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'

