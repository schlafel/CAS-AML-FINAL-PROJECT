import sys

sys.path.insert(0, '..')
from config import *
from augmentations import *

import os
import pandas as pd
import numpy as np
import random
import json

import tensorflow as tf

# Label dictionaries
label_dict = json.load(open(os.path.join(ROOT_PATH,RAW_DATA_DIR,MAP_JSON_FILE)))
label_dict_inference = dict(zip(label_dict.values(),label_dict.keys()))
class ASL_DATASET:
    """
    A dataset class for the ASL dataset.
    """

    # Constructor method
    def __init__(self, metadata_df=None, transform=None, max_seq_length=INPUT_SIZE, augment=False, augmentation_threshold = .1):
        """
        Initialize the dataset.

        Args:
            metadata_df (pd.DataFrame, optional): A dataframe containing the metadata for the dataset.
            transform (callable, optional): A function/transform to apply to the data.
            max_seq_length (int): The maximum sequence length for the data.
            augment (bool): Whether to apply data augmentation. The augmentation_threshold parameter defines at which
            probability the transformation will happen.

        Functionality:
            Constructor method

        :rtype: object
        :param metadata_df: A dataframe containing the metadata for the dataset.
        :param transform: A function/transform to apply to the data.
        :type transform: callable, optional
        :param max_seq_length: The maximum sequence length for the data.
        :type max_seq_length: int
        :param augment: Whether to apply data augmentation.
        :type augment: bool
        :param augmentation_threshold: Probability of augmentation happening. Only if augment == True
        :type augmentation_threshold: float
        """

        super().__init__()

        self.transform = transform
        self.augment = augment
        self.augmentation_threshold = augmentation_threshold

        # [TODO] get this from data
        self.max_seq_length = max_seq_length

        self.df_train = metadata_df

        self.load_data()

    # Load the data method
    def load_data(self):
        """
        Load the data for the dataset.

        Functionality:
            Load the data method

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

    # Get the length of the dataset
    def __len__(self):
        """
        Return the length of the dataset.

        Functionality:
            Get the length of the dataset

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

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the landmarks, target, and size for the item.

        Functionality:
            Get a single item from the dataset

        :param idx: The index of the item to retrieve.
        :type idx: int
        :return: A dictionary containing the landmarks, target, and size for the item.
        :rtype: dict
        """
        #if torch.is_tensor(idx):
        #    idx = idx.item()

        # Get the processed data for the single index
        landmark_file = self.file_paths[idx]

        # Read in the processed file
        sample = np.load(landmark_file)
        landmarks = sample["landmarks"]
        sample.close()
        #dmarks = sample["landmarks"]
        # Get the processed landmarks and target for the data
        target = self.target[idx]
        size = self.size[idx]

        if self.transform:
            landmarks = self.transform(landmarks)

        if self.augment:
            if random.random() < self.augmentation_threshold:
                landmarks = shift_landmarks(landmarks)
            if random.random() < self.augmentation_threshold:
                landmarks = mirror_landmarks(landmarks)  # TODO : Change Names
            if random.random() < self.augmentation_threshold:
                landmarks = random_scaling(landmarks)
            if random.random() < self.augmentation_threshold:
                landmarks = random_rotation(landmarks)  # TODO : Change Names
            if random.random() < self.augmentation_threshold:
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

        Returns:
            str: A string representation of the dataset.

        Functionality:
            Return a string representation of the dataset

        :return: A string representation of the dataset.
        :rtype: str
        """

        return f'ASL_DATASET(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'


class ASL_DATASET_TF(ASL_DATASET):
    """
    Subclass of ASL_DATASET to get the parent class working also in tensorflow...
    """
    def __init__(self,metadata_df=None, transform=None, max_seq_length=INPUT_SIZE, augment=False):
        super().__init__(metadata_df=metadata_df, transform=transform, max_seq_length=max_seq_length, augment=augment)

    def create_dataset(self,
                       batch_size:int = 32,
                       shuffle = False,
                       augment = False
                       ):
        """
        Main Method to get the TensorFlow-Dataset

        :param batch_size: Batchsize
        :type batch_size: int
        :param shuffle: Shuffle the dataset
        :type shuffle: bool
        :param augment: Augment the dataset
        :type augment: bool
        :return: dataset tf.data.Dataset

        """

        def augment_fn(x, y, augmentation_threshold = self.augmentation_threshold):
            # tf.print("Augmenting Data!!! ")
            # x = x
            if tf.random.uniform([]) > augmentation_threshold:
                [x, ] = tf.py_function(random_scaling, [x,(0.9,2)], [tf.float32])
            if tf.random.uniform([]) > augmentation_threshold:
                [x, ] = tf.py_function(random_rotation, [x], [tf.float32])
            if tf.random.uniform([]) > augmentation_threshold:
                [x, ] = tf.py_function(mirror_landmarks, [x], [tf.float32])
            # if tf.random.uniform([]) > augmentation_threshold:
            #     [x, ] = tf.py_function(frame_dropout, [x], [tf.float32])

            return x, y

        def load_data(x, y):
            def load_data_np(fp):
                # load the data
                sample = np.load(fp.numpy())

                return sample.astype(np.float32)

            x = tf.py_function(load_data_np, inp=[x], Tout=tf.float32)
            # x = tf.squeeze(x,axis = 0)

            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((self.file_paths,
                                                       self.target.astype(np.int32)))
        if shuffle:
            dataset = dataset.shuffle(len(self))

        dataset = dataset.map(load_data,
                              num_parallel_calls=tf.data.AUTOTUNE)

        # do the augmentation
        if augment:
            dataset = dataset.map(augment_fn)

        dataset = dataset.cache()

        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def __repr__(self):
        """
        Return a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.

        Functionality:
            Return a string representation of the dataset

        :return: A string representation of the dataset.
        :rtype: str
        """

        return f'ASL_DATASET_TF(Participants: {len(set(self.participant_ids))}, Length: {len(self.df_train)}'



if __name__ == '__main__':

    dataset = ASL_DATASET(augment=True)

    tf_dataset = ASL_DATASET_TF(augment=True).create_dataset()


    for X,y in tf_dataset:
        print(X.shape)


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
