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
    def __init__(self, df, batch_size=32,
                 max_seq_length=MAX_SEQUENCES,
                 N_LANDMARKS=N_LANDMARKS,
                 N_DIMS = len(COLUMNS_TO_USE),
                 augment=True,
                 augmentation_threshold=.5,
                 ):


        self.df = df
        self.batch_size = batch_size

        self.augment = augment
        self.augmentation_threshold = augmentation_threshold

        self.path = self.df.full_path.values
        self.target = self.df.target.values.astype(np.int64)

        self.max_seq_length = max_seq_length
        self.df = df
        self.out_shape = (max_seq_length,N_LANDMARKS,N_DIMS)


        self.augmentation_threshold = .5

    def __getitem__(self, idx):
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
                       sample = None,
                       batch_size=BATCH_SIZE,
                       max_seq_length=MAX_SEQUENCES,
                       N_LANDMARKS=N_LANDMARKS,
                       N_DIMS=len(COLUMNS_TO_USE),
                       augment=True,
                       augmentation_threshold=.5,
                       ):
    df = pd.read_csv(csv_path)
    df["full_path"] = df.path.apply(map_full_path,args=(data_path,))
    
    if sample is not None:
        df = df.sample(sample)
    
    #Get the dataset
    cDM = ASL_TF_DATASET(df,
                         batch_size=batch_size,
                         max_seq_length=max_seq_length,
                         N_LANDMARKS=N_LANDMARKS,
                         N_DIMS=N_DIMS,
                         augment=augment,
                         augmentation_threshold=augmentation_threshold,
                         )


    dataset = cDM.get_dataset(batch_size=240,
                              shuffle_buffer_size=250,
                              )



    return dataset



if __name__ == '__main__':


    dataset = get_TF_ASL_DATASET()
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

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(150,184,2,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(90, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    model.fit(dataset,
              epochs = 15,
              #validation_split=.1
              #steps_per_epoch=n_steps,
              )