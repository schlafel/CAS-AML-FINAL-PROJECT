"""
=======================
ASL Dataset description
=======================

This file contains the ASL_DATASET class which serves as the dataset module for American Sign Language (ASL) data.
The ASL_DATASET is designed to load, preprocess, augment, and serve the dataset for model training and validation.
This class provides functionalities such as loading the dataset from disk, applying transformations, data augmentation
techniques, and an interface to access individual data samples.

.. note::
    This dataset class expects data in a specific format. Detailed explanations and expectations about input data are
    provided in respective method docstrings.
"""
import sys

sys.path.insert(0, '..')
from config import *
from augmentations import normalize, standardize, random_rotation, random_scaling, frame_dropout, mirror_landmarks, shift_landmarks

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

    The ASL_DATASET class represents a dataset of American Sign Language (ASL) gestures, where each gesture corresponds to a
    word or phrase. This class provides functionalities to load the dataset, apply transformations, augment the data,
    and yield individual data samples for model training and validation.
    """
    # Constructor method
    def __init__(self, metadata_df=None, transform=None,
                 max_seq_length=INPUT_SIZE,
                 augment=False,
                 augmentation_threshold=0.1,
                 enableDropout=True):
        """
        Initialize the ASL dataset.

        This method initializes the dataset and loads the metadata necessary for the dataset processing.
        If no metadata is provided, it will load the default processed dataset.
        It also sets the transformation functions, data augmentation parameters, and maximum sequence length.

        Args:
            metadata_df (pd.DataFrame, optional): A dataframe containing the metadata for the dataset. Defaults to None.
            transform (callable, optional): A function/transform to apply to the data. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length for the data. Defaults to INPUT_SIZE.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            augmentation_threshold (float, optional): Probability of augmentation happening. Only if augment == True. Defaults to 0.1.
            enableDropout (bool, optional): Whether to enable the frame dropout augmentation. Defaults to True.

        Functionality:
            Initializes the dataset with necessary configurations and loads the data.

        :param metadata_df: A dataframe containing the metadata for the dataset.
        :type metadata_df: pd.DataFrame, optional
        :param transform: A function/transform to apply to the data.
        :type transform: callable, optional
        :param max_seq_length: The maximum sequence length for the data.
        :type max_seq_length: int
        :param augment: Whether to apply data augmentation.
        :type augment: bool
        :param augmentation_threshold: Probability of augmentation happening. Only if augment == True.
        :type augmentation_threshold: float
        :param enableDropout: Whether to enable the frame dropout augmentation.
        :type enableDropout: bool
        """

        super().__init__()

        self.transform = transform
        self.augment = augment
        self.augmentation_threshold = augmentation_threshold
        self.enableDropout = enableDropout

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

        self.df_train = metadata_df

        self.load_data()

    # Load the data method
    def load_data(self):
        """
        Load the data for the ASL dataset.

        This method loads the actual ASL data based on the metadata provided during initialization. If no metadata was
        provided, it loads the default processed data. It generates absolute paths to locate landmark files, and
        stores individual metadata lists for easy access during data retrieval.

        Functionality:
            Loads the data for the dataset.

        :rtype: None
        """

        if self.df_train is None:
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

        #create target_dict for conversion
        df = self.df_train[['target','sign']].drop_duplicates().sort_values(by="target")
        self.target_dict = dict(zip(df['target'],df["sign"]))

    # Get the length of the dataset
    def __len__(self):
        """
        Get the length of the dataset.

        This method returns the total number of data samples present in the dataset. It's an implementation of the
        special method __len__ in Python, providing a way to use the Python built-in function len() on the dataset object.

        Functionality:
            Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.df_train)

    # Get a single item from the dataset
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        This method returns a data sample from the dataset based on a provided index. It handles reading of the processed
        data file, applies transformations and augmentations (if set), and pads the data to match the maximum sequence length.
        It returns the preprocessed landmarks and corresponding target as a tuple.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the landmarks and target for the item.

        Functionality:
            Get a single item from the dataset.

        :param idx: The index of the item to retrieve.
        :type idx: int
        :return: A tuple containing the landmarks and target for the item.
        :rtype: tuple
        """
        #if torch.is_tensor(idx):
        #    idx = idx.item()

        # Get the processed data for the single index
        landmark_file = self.file_paths[idx]

        # Read in the processed file
        landmarks = np.load(landmark_file).astype('float32')

        # Get the processed landmarks and target for the data
        target = self.target[idx]
        size = self.size[idx]

        if self.transform:
            landmarks = self.transform(landmarks)

        if self.augment:
            if random.random() < self.augmentation_threshold:
                landmarks = shift_landmarks(landmarks)
            if random.random() < self.augmentation_threshold:
                landmarks = mirror_landmarks(landmarks)
            if random.random() < self.augmentation_threshold:
                landmarks = random_scaling(landmarks)
            if random.random() < self.augmentation_threshold:
                landmarks = random_rotation(landmarks)
            if random.random() < self.augmentation_threshold and self.enableDropout:
                landmarks = frame_dropout(landmarks)
                size = len(landmarks)

        # Pad the landmark data
        pad_len = max(0, self.max_seq_length - len(landmarks))

        padding = np.zeros((pad_len, landmarks.shape[1], landmarks.shape[2]))
        landmarks = np.vstack([landmarks, padding]).astype('float32')

        landmarks = standardize(landmarks)

        sample = (landmarks, target)

        return sample

    # Return a string representation of the dataset
    def __repr__(self):
        """
        Return a string representation of the ASL dataset.

        This method returns a string that provides an overview of the dataset, including the number of participants and
        total data samples. It's an implementation of the special method __repr__ in Python, providing a human-readable
        representation of the dataset object.

        Returns:
            str: A string representation of the dataset.

        Functionality:
            Return a string representation of the dataset.

        :return: A string representation of the dataset.
        :rtype: str
        """

        return f'ASL_DATASET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'



if __name__ == '__main__':

    dataset = ASL_DATASET(augment=True)

    print(dataset)
    
    
    
