import sys

sys.path.insert(0, '..')
from config import *

import tensorflow as tf
from torch.utils.data import DataLoader
import math
import yaml

def get_model_params(model_name):
    with open('models/modelconfig.yaml') as f:
        data = yaml.safe_load(f)
    return data['models'][model_name]

def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, dl_framework=DL_FRAMEWORK,num_workers=os.cpu_count()):
    
    if dl_framework=='tensorflow':
        return to_TF_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return to_PT_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)

class DatasetWithLen:
    def __init__(self, tf_dataset, length):
        self.tf_dataset = tf_dataset
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.tf_dataset)

def to_TF_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    def preprocess_sample(landmark, target):
        if not tf.is_tensor(landmark):
            landmark = tf.convert_to_tensor(landmark)
        if not tf.is_tensor(target):
            target = tf.convert_to_tensor(target)
        return landmark, target

    def shuffled_dataset_gen():
        indices = np.arange(len(dataset))

        if shuffle:
            np.random.shuffle(indices)

        for index in indices:
            yield dataset[index]


    tf_data = tf.data.Dataset.from_generator(shuffled_dataset_gen, output_types=(tf.float32, tf.int32))

    #tf_data = tf_data.take(batch_size).cache()

    tf_data = tf_data.map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    tf_data = tf_data.batch(batch_size)

    dataset_with_len = DatasetWithLen(tf_data, math.ceil(len(dataset) / batch_size))
    return dataset_with_len


def to_PT_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers = os.cpu_count()):
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def log_metrics(phase, loss, acc, epoch, lr, writer):
        """Helper function to log metrics to TensorBoard"""
        writer.add_scalar(f'Loss/{phase}', loss, epoch+1)
        writer.add_scalar(f'Accuracy/{phase}', acc, epoch+1)
        print(f"EPOCH {epoch + 1:>3}: {phase} accuracy: {acc:>3.2f}, {phase} Loss: {loss:>9.8f}, LRate {lr:>9.8f} ",
              flush=True)
