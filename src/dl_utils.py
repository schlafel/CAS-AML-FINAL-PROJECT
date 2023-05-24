import sys

sys.path.insert(0, '..')
from config import *

import tensorflow as tf
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, dl_framework=DL_FRAMEWORK,num_workers=os.cpu_count()):
    
    if dl_framework=='tensorflow':
        return to_TF_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return to_PT_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)

def to_TF_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    def preprocess_sample(landmark, target):
        return tf.constant(landmark), tf.constant(target)
            
    tf_data = tf.data.Dataset.from_generator(lambda: dataset, output_types=(tf.float32, tf.int32))    
        
    if shuffle:
        tf_data = tf_data.shuffle(buffer_size=len(dataset))
    
    tf_data = tf_data.cache()
    
    tf_data = tf_data.map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    tf_data = tf_data.batch(batch_size)
    return tf_data
    
def to_PT_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers = os.cpu_count()):
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    