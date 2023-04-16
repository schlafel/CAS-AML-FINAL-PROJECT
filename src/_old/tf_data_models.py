
import pandas as pd

from config import *
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import List, Tuple
import json
import numpy as np
import tensorflow_probability as tfp

class MyGenerator:
    def __init__(self, filenames: List[str], labels: List[int], batch_size: int,  shuffle: bool):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.filenames)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.current_index >= self.num_samples:
            if self.shuffle:
                arr_in = np.array(list(zip(self.filenames, self.labels)))
                np.random.shuffle(arr_in)
                self.labels = arr_in[:,1]
                self.filenames = arr_in[:,0]

            self.current_index = 0
        batch_filenames = self.filenames[self.current_index:self.current_index + self.batch_size]
        batch_labels = self.labels[self.current_index:self.current_index + self.batch_size]
        batch_data = []
        for filename,label in zip(batch_filenames,batch_labels):
            # Load the data from disk and preprocess it
            data = self.load_and_preprocess_data(filename,label)
            batch_data.append(data)
        batch_x, batch_y = zip(*batch_data)
        batch_x = tf.stack(batch_x)
        batch_y = tf.stack(batch_y)
        self.current_index += self.batch_size
        return batch_x, batch_y

    def load_and_preprocess_data(self, filename: str,label:int) -> Tuple[tf.Tensor, tf.Tensor]:
        # Load the data from disk and preprocess it
        # In this example, we simply return the input tensors as-is
        df_in = pd.read_parquet(filename)

        #get only interesting landmarks
        tens_xy = tf.convert_to_tensor(df_in[["x","y"]].fillna(0).values)
        seq_length = tens_xy.shape[0]//N_LANDMARKS

        #reshape the tensor
        res_tf = tf.reshape(tens_xy, (seq_length,  N_LANDMARKS,2))


        #extrakt landmarks
        sliced_tf = tf.gather(res_tf,LANDMARK_INDICES,axis = 1)

        # # now interpolate
        #
        # # create a mask to identify missing values (including 0)
        # mask = tf.cast(tf.not_equal(sliced_tf, 0), tf.float32)
        #
        # # compute the indices of the missing values (including 0)
        # indices = tf.squeeze(tf.where(tf.equal(sliced_tf, 0)))
        #
        # # create a regular grid for interpolation
        # grid = tf.linspace(0., tf.cast(tf.size(sliced_tf) - 1, tf.float32), tf.size(sliced_tf))
        #
        # # interpolate the missing values (including 0)
        # interpolated_values = tfp.math.interp_regular_1d_grid(
        #     x=indices,
        #     x_ref=grid,
        #     y_ref=tf.boolean_mask(sliced_tf, mask),
        #     axis=0,
        #     fill_value='extrapolate'
        # )
        #
        # # replace the missing values (including 0) with the interpolated values
        # sliced_tf_interp = tf.tensor_scatter_nd_update(sliced_tf, tf.expand_dims(indices, axis=1), interpolated_values)





        # pad
        # Define the amount of padding
        if seq_length>MAX_SEQUENCES:
            padded_tf = tf.slice(sliced_tf, [0, 0,0], [MAX_SEQUENCES, -1, -1])
        else:
            padding = [[MAX_SEQUENCES-seq_length, 0], [0, 0], [0, 0]]

            #apply the padding...
            padded_tf = tf.pad(sliced_tf,padding)

        #finally: reshape

        flat_tf = tf.reshape(padded_tf,-1)


        return flat_tf, tf.constant(label, shape=(1, ))


class MyDataModule(tf.keras.utils.Sequence):
    def __init__(self, path_in, path_json, batch_size: int,  shuffle: bool, validation_split: float):
        self.setup(path_in,path_json)
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.validation_split = validation_split

        idx = np.array(list(range(0, len(self.labels))))
        np.random.shuffle(idx)
        # Split the data into training and validation sets
        split_idx = int(len(self.filenames) * (1.0 - self.validation_split))
        train_filenames = list(np.array(self.filenames)[idx][:split_idx])
        train_labels = list(np.array(self.labels)[idx][:split_idx])
        val_filenames = list(np.array(self.filenames)[idx][split_idx:])
        val_labels = list(np.array(self.labels)[idx][split_idx:])

        # Create the data loaders for training and validation sets
        self.train_loader = MyGenerator(train_filenames,  train_labels,batch_size, shuffle)
        self.val_loader = MyGenerator(val_filenames,  val_labels,  batch_size,shuffle)
    def setup(self,path_in,path_json):
        df_in = pd.read_csv(path_in)
        df_in = df_in.iloc[0:100]

        self.label_dict = json.load(open(path_json))
        #read json map



        filenames = [os.path.join(ROOT_PATH,RAW_DATA_DIR,row.path) for _i,row in df_in.iterrows()]
        self.labels = df_in.sign.map(self.label_dict).values

        self.filenames = filenames
    def __len__(self) -> int:
        # Return the total number of batches in the dataset
        return len(self.train_loader)

    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get the batch at the specified index


        batch_x, batch_y = self.train_loader.__next__()
        return batch_x, batch_y

    def get_validation_data(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get the validation data
        batch_x, batch_y = next(iter(self.val_loader))
        return batch_x, batch_y


class TF_ASL_Dataset(tf.keras.utils.Sequence):
    def __init__(self, path_train, batch_size,MAX_SEQUENCES = MAX_SEQUENCES, *args, **kwargs):
        self.batch_size = batch_size
        self.path_train = path_train
        #self.train_data = np.random.random((70,2))
        self.setup()
        self.n_items = self.train_data.shape[0]


        self.max_sequences = MAX_SEQUENCES
    def setup(self):
        df_in = pd.read_csv(self.path_train)
        path_json = os.path.join(ROOT_PATH, RAW_DATA_DIR, MAP_JSON_FILE)

        self.label_dict = json.load(open(path_json))



        df_in["full_path"] = [os.path.join(ROOT_PATH,RAW_DATA_DIR,row.path) for _i, row in df_in.iterrows()]
        df_in["labels"] = df_in.sign.map(self.label_dict)



        self.train_data = df_in[["full_path","labels"]].values

        pass
    def __len__(self):
        # returns the number of batches

        return int(np.ceil(self.n_items / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.n_items)
        # returns one batch
        batch = self.train_data[low:high,:]
        file_names = batch[:,0]
        signs = batch[:,1]
        y = tf.convert_to_tensor(signs,dtype=tf.int16)[...,tf.newaxis]

        for _i,filename in enumerate(file_names):
            df_in = pd.read_parquet(filename)

            #get only interesting landmarks
            tens_xy = tf.convert_to_tensor(df_in[["x","y"]].fillna(0).values)
            seq_length = tens_xy.shape[0]//N_LANDMARKS

            #reshape the tensor
            res_tf = tf.reshape(tens_xy, (seq_length,  N_LANDMARKS,2))

            #extrakt landmarks
            sliced_tf = tf.gather(res_tf,LANDMARK_INDICES,axis = 1)

            # Reshape to dimension (seq_length,2*len(Lanmark_indices)
            sliced_tf = tf.reshape(sliced_tf, [seq_length, len(LANDMARK_INDICES) * 2])

            # Define the amount of padding
            if seq_length > self.max_sequences:
                padded_tf = tf.slice(sliced_tf, [0, 0,], [self.max_sequences, -1, ])
            else:
                padding = [[self.max_sequences - seq_length, 0], [0, 0]]
                # apply the padding...
                padded_tf = tf.pad(sliced_tf, padding)

            # finally: reshape

            #flat_tf = tf.reshape(padded_tf, -1)

            #stack to tensor
            if _i == 0:
                X = padded_tf[tf.newaxis,...]
            else:
                X = tf.concat([X,(padded_tf[tf.newaxis,...])],axis = 0)



        return X, y

    def on_epoch_end(self):
        # option method to run some logic at the end of each epoch: e.g. reshuffling
        np.random.shuffle(self.train_data)


if __name__ == '__main__':
    np.random.seed(42)
    MAX_SEQUENCES = 150
    cds = TF_ASL_Dataset(batch_size=64,MAX_SEQUENCES = MAX_SEQUENCES)

    print("done")
    for batch in cds:
        X,y = batch
        print(X.shape,y.shape)




    path_train_file = os.path.join(ROOT_PATH,RAW_DATA_DIR,TRAIN_CSV_FILE)
    path_json = os.path.join(ROOT_PATH,RAW_DATA_DIR,MAP_JSON_FILE)

    dM = MyDataModule(path_in = path_train_file,path_json = path_json,
                      batch_size = 64, shuffle=True,
                      validation_split=0.2)

    next(iter(dM))

    for batch in dM.train_loader:
        landmarks,labels = batch
        print(landmarks.shape,labels.shape)

    print("Second epoch")
    #yolo next epoch
    for batch in dM.train_loader:
        landmarks,labels = batch
        print(landmarks.shape,labels.shape)

    print("Ended second epoch")

    print("done")