import os.path

import pandas as pd
import tensorflow as tf
from src.config import *
from src.augmentations import *
import numpy as np
import time
from tqdm import tqdm
import json
import matplotlib as mpl
import matplotlib.pyplot as plt



class ASL_TF_DATASET():
    def __init__(self, df,
                 max_seq_length=MAX_SEQUENCES,
                 N_LANDMARKS=N_LANDMARKS,
                 N_DIMS = len(COLUMNS_TO_USE),
                 augment=True,
                 augmentation_threshold=.5,
                 ):


        self.df = df

        self.augment = augment
        self.augmentation_threshold = augmentation_threshold

        self.path = self.df.full_path.values
        self.target = self.df.target.values.astype(np.int64)

        self.max_seq_length = max_seq_length
        self.df = df
        self.out_shape = (max_seq_length,N_LANDMARKS,N_DIMS)


        self.augmentation_threshold = .5

    def __getitem__(self, idx):
        #load the data
        sample = np.load(self.path[idx])
        target = np.array([self.target[idx]])

        landmarks = self.pad_sequence(sample)

        return landmarks, target

    def pad_sequence(self, sample):
        # pad the sequence
        if sample.shape[0] < self.max_seq_length:
            landmarks = np.append(sample, np.zeros(((self.max_seq_length - sample.shape[0]), sample.shape[1], 2)),
                                  axis=0)
        else:
            # crop
            landmarks = sample[:self.max_seq_length, :, :]
        return landmarks

    def data_generator(self):
        # Generator function that yields data for the dataset
        for idx in range(len(self)):
            X, y = self.__getitem__(idx)
            yield X, y

    def get_dataset(self, batch_size=BATCH_SIZE, shuffle_buffer_size=None,):
        # Create a dataset using tf.data.Dataset.from_generator
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=self.out_shape, dtype=tf.float64),
                tf.TensorSpec(shape=(1,), dtype=tf.int64)
            )
        )

        #cache the dataset
        dataset = dataset.cache()

        #augment data
        if self.augment:
            dataset = dataset.map(self.augment_data,
                                  num_parallel_calls=tf.data.AUTOTUNE)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    # Define the data augmentation function
    def augment_data(self, x, y):
        #Get the original shape....
        x_shape = x.shape
        y_shape = y.shape

        if tf.random.uniform([]) > self.augmentation_threshold:
            [x, ] = tf.py_function(random_scaling, [x], [tf.float64])
        if tf.random.uniform([]) >  self.augmentation_threshold:
            [x, ] = tf.py_function(random_rotation, [x], [tf.float64])
        if tf.random.uniform([]) >  self.augmentation_threshold:
            [x, ] = tf.py_function(mirror_landmarks2, [x], [tf.float64])
        if tf.random.uniform([]) >  self.augmentation_threshold:
            [x, ] = tf.py_function(shift_landmarks2, [x], [tf.float64])


        #Reset the shape
        #x.set_shape(x_shape)
        #y.set_shape(y_shape)

        return x, y
    def __len__(self):
        return (self.df.shape[0])


# Get complete file path to file
def map_full_path(path,data_path = os.path.join(ROOT_PATH,PROCESSED_DATA_DIR)):
    return os.path.join(data_path, path)

def get_TF_ASL_DATASET(csv_path = os.path.join(ROOT_PATH,PROCESSED_DATA_DIR,TRAIN_CSV_FILE),
                      data_path = os.path.join(ROOT_PATH,PROCESSED_DATA_DIR),
                       sample:int = None,
                       batch_size:int=BATCH_SIZE,
                       max_seq_length:int=MAX_SEQUENCES,
                       N_LANDMARKS:int=N_LANDMARKS,
                       N_DIMS:int=len(COLUMNS_TO_USE),
                       augment:bool=True,
                       augmentation_threshold:float=.5,
                       ):
    """
    Training-Pipeline
    :param csv_path: Path to the Training CSV-File (Can be Processed or not)
    :param data_path: Path to the base data
    :param sample: Optional --> For testing (ilmits the number of items to read)
    :param max_seq_length: Maximum sequence length
    :param N_LANDMARKS: Number of landmarks used
    :param N_DIMS: Number of Dimensions used
    :param augment: Wheter to augment or not
    :param augmentation_threshold: Threshold for augmentation for each sequence 1 --> Every item gets augmented
    :return: tf.data.Dataset for ASL-DATA based on input csv-File
    """
    df = pd.read_csv(csv_path)
    df["full_path"] = df.path.apply(map_full_path,args=(data_path,))
    
    if sample is not None:
        df = df.sample(sample)
    
    #Get the dataset
    cDM = ASL_TF_DATASET(df,
                         max_seq_length=max_seq_length,
                         N_LANDMARKS=N_LANDMARKS,
                         N_DIMS=N_DIMS,
                         augment=augment,
                         augmentation_threshold=augmentation_threshold,
                         )


    dataset = cDM.get_dataset(batch_size=batch_size,
                              shuffle_buffer_size=250,
                              )



    return dataset


def pad_sequence(sample,max_seq_length = MAX_SEQUENCES):
    # pad the sequence
    if sample.shape[0] < max_seq_length:
        landmarks = np.append(sample, np.zeros(((max_seq_length - sample.shape[0]), sample.shape[1], 2)),
                              axis=0)
    else:
        # crop
        landmarks = sample[:max_seq_length, :, :]
    return landmarks

def augment(x,y, augmentation_threshold = .5):
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
        return frames.numpy() * scale_factor

    if tf.random.uniform([]) > augmentation_threshold:
        [x, ] = tf.py_function(random_scaling, [x], [tf.float32])
    # # if tf.random.uniform([]) > augmentation_threshold:
    # #     [x, ] = tf.py_function(random_rotation, [x], [tf.float64])
    # if tf.random.uniform([]) > augmentation_threshold:
    #     [x, ] = tf.py_function(mirror_landmarks2, [x], [tf.float64])
    # if tf.random.uniform([]) > augmentation_threshold:
    #     [x, ] = tf.py_function(shift_landmarks2, [x], [tf.float64])

    return x,y


def load_data(x,y):
    def load_data_np(fp):
        # load the data
        sample = np.load(fp.numpy())

        landmarks = pad_sequence(sample)

        return landmarks.astype(np.float32)

    x = tf.py_function(load_data_np,inp = [x],Tout=tf.float32)
    # x = tf.squeeze(x,axis = 0)

    return x,y

def get_tf_dataset(csv_path,
                   data_path,
                   batch_size = 250,
                   augment_data = True):

    df = pd.read_csv(csv_path)
    df["full_path"] = df.path.apply(map_full_path, args=(data_path,))

    dataset1 = tf.data.Dataset.from_tensor_slices((df.full_path.values,
                                                   df.target.values.astype(np.int32)))
    dataset = dataset1.shuffle(len(df))

    dataset = dataset.map(load_data,
                          num_parallel_calls=tf.data.AUTOTUNE)

    # do the augmentation
    dataset = dataset.map(augment)

    dataset = dataset.cache()


    dataset = dataset.batch(batch_size)


    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    return dataset

if __name__ == '__main__':
    csv_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE)
    data_path = os.path.join(ROOT_PATH, PROCESSED_DATA_DIR)


    df = pd.read_csv(csv_path)
    df["full_path"] = df.path.apply(map_full_path, args=(data_path,))


    dataset = get_tf_dataset(csv_path,
                   data_path,
                   batch_size = 250)
    #
    # dataset1 = tf.data.Dataset.from_tensor_slices((df.full_path.values,
    #                                                df.target.values))
    # dataset = dataset1.shuffle(len(df))
    #
    # dataset = dataset.map(load_data,
    #                       num_parallel_calls=tf.data.AUTOTUNE)
    #
    #
    # #do the augmentation
    # dataset = dataset.map(augment)
    # dataset = dataset.cache()
    #
    # dataset = dataset.batch(250)
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    t0 = time.time()
    for batchX,batchY in tqdm(dataset):
        break
    print(time.time()-t0)


    #dataset = get_TF_ASL_DATASET(batch_size=32)
    # df = pd.read_csv(os.path.join(ROOT_PATH, PROCESSED_DATA_DIR, TRAIN_CSV_FILE))
    # df["full_path"] = df.path.apply(map_full_path)
    # cDM = ASL_TF_DATASET(df.sample(1000), )
    # X, y = next(iter(cDM))
    # print(X.shape)
    # print(len(cDM))

    #dataset = cDM.get_dataset(batch_size=250, shuffle_buffer_size=1)



    # t0 = time.time()
    # n_steps = 0
    # print("Running through the dataset:", time.time())
    # for batch_X, batch_y in tqdm(dataset):
    #     #print(batch_X.shape, batch_y.shape)
    #     #print(n_steps)
    #     n_steps+= 1
    #     pass
    #
    # print(n_steps, time.time() - t0)
    # print(batch_X.shape  )

    #### Run it through a mlp
    input_shape = (MAX_SEQUENCES,N_LANDMARKS,len(COLUMNS_TO_USE))
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((150, N_LANDMARKS*2), input_shape=input_shape),
        tf.keras.layers.LSTM(64,return_sequences = True),
        tf.keras.layers.LSTM(128,return_sequences = True),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(75, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    model.fit(dataset,
              epochs = 150,
              #validation_split=.1
              #steps_per_epoch=n_steps,
              )