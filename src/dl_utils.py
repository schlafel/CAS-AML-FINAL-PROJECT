"""
===================
Deep Learning Utils
===================

This module provides a set of helper functions that abstract away specific details of different deep learning frameworks
(such as TensorFlow and PyTorch). These functions allow the main code to run in a framework-agnostic manner,
thus improving code portability and flexibility.

"""
import sys

sys.path.insert(0, '..')
from config import *

import tensorflow as tf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter,summary
import math
import yaml


def get_model_params(model_name):
    """
    The get_model_params function is a utility function that serves to abstract away the details of reading model
    configurations from a YAML file. In a machine learning project, it is common to have numerous models, each with its
    own set of hyperparameters. These hyperparameters can be stored in a YAML file for easy access and modification.

    This function reads the configuration file and retrieves the specific parameters associated with the given model.
    The configurations are stored in a dictionary which is then returned. This aids in maintaining a cleaner, more
    organized codebase and simplifies the process of updating or modifying model parameters.

    Args:
        model_name: Name of the model whose parameters are to be retrieved.

    Functionality:
        Reads a YAML file and retrieves the model parameters as a dictionary.

    :rtype: dict
    :param model_name: Name of the model whose parameters are to be retrieved.
    """
    with open(os.path.join(ROOT_PATH, SRC_DIR, MODEL_DIR, 'modelconfig.yaml')) as f:
        data = yaml.safe_load(f)
    return data['models'][model_name]


def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, dl_framework=DL_FRAMEWORK, num_workers=os.cpu_count()):
    """
    The get_dataloader function is responsible for creating a DataLoader object given a dataset and a few other
    parameters. A DataLoader is an essential component in machine learning projects as it controls how data is fed into
    the model during training. However, different deep learning frameworks have their own ways of creating and handling
    DataLoader objects.

    To improve the portability and reusability of the code, this function abstracts away these specifics, allowing the
    user to create a DataLoader object without having to worry about the details of the underlying framework
    (TensorFlow or PyTorch). This approach can save development time and reduce the risk of bugs or errors.

    Args:
        dataset: The dataset to be loaded.
        batch_size: The size of the batches that the DataLoader should create.
        shuffle: Whether to shuffle the data before creating batches.
        dl_framework: The name of the deep learning framework.
        num_workers: The number of worker threads to use for loading data.

    Functionality:
        Creates and returns a DataLoader object that is compatible with the specified deep learning framework.

    :rtype: DataLoader or DatasetWithLen object
    :param dataset: The dataset to be loaded.
    :param batch_size: The size of the batches that the DataLoader should create.
    :param shuffle: Whether to shuffle the data before creating batches.
    :param dl_framework: The name of the deep learning framework.
    :param num_workers: The number of worker threads to use for loading data.
    """
    if dl_framework == 'tensorflow':
        return to_TF_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return to_PT_DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class DatasetWithLen:
    """
    The DatasetWithLen class serves as a wrapper around TensorFlow's Dataset object. Its primary purpose is to add a
    length method to the TensorFlow Dataset. This is useful in contexts where it's necessary to know the number of
    batches that a DataLoader will create from a dataset, which is a common requirement in many machine learning
    training loops. It also provides an iterator over the dataset, which facilitates traversing the dataset for
    operations such as batch creation.

    For instance, this might be used in conjunction with a progress bar during training to display the total number of
    batches. Since TensorFlow's Dataset objects don't inherently have a __len__ method, this wrapper class provides that
    functionality, augmenting the dataset with additional features that facilitate the training process.

    Args:
        tf_dataset: The TensorFlow dataset to be wrapped.
        length: The length of the dataset.

    Functionality:
        Provides a length method and an iterator for a TensorFlow dataset.

    :rtype: DatasetWithLen object
    :param tf_dataset: The TensorFlow dataset to be wrapped.
    :param length: The length of the dataset.
    """
    def __init__(self, tf_dataset, length):
        self.tf_dataset = tf_dataset
        self.length = length
        self.augmentation_threshold = 0.1

    def __len__(self):
        """
        Returns the length of the dataset.

        :return: length of the dataset
        """
        return self.length

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        :return: iterator for the dataset
        """
        return iter(self.tf_dataset)


def to_TF_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    """
    This function takes in a dataset and converts it into a TensorFlow DataLoader. Its purpose is to provide a
    streamlined method to generate DataLoaders that can be utilized in a TensorFlow training or inference pipeline.
    It not only ensures the dataset is in a format that can be ingested by TensorFlow's pipeline, but also implements
    optional shuffling of data, which is a common practice in model training to ensure random distribution of data
    across batches.

    This function first checks whether the data is already in a tensor format, if not it converts the data to a
    tensor. Next, it either shuffles the dataset or keeps it as is, based on the 'shuffle' flag. Lastly, it prepares
    the TensorFlow DataLoader by batching the dataset and applying an automatic optimization strategy for the number
    of parallel calls in mapping functions.

    Args:
        dataset: The dataset to be loaded.
        batch_size: The size of each batch the DataLoader will return.
        shuffle: Whether the data should be shuffled before batching.

    Functionality:
        Converts a given dataset into a TensorFlow DataLoader.

    :rtype: DatasetWithLen object
    :param dataset: The dataset to be loaded.
    :param batch_size: The size of each batch the DataLoader will return.
    :param shuffle: Whether the data should be shuffled before batching.
    """
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

    tf_data = tf_data.cache()

    tf_data = tf_data.map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
    tf_data = tf_data.batch(batch_size)

    dataset_with_len = DatasetWithLen(tf_data, math.ceil(len(dataset) / batch_size))
    return dataset_with_len


def to_PT_DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()):
    """
    This function is the PyTorch counterpart to 'to_TF_DataLoader'. It converts a given dataset into a PyTorch
    DataLoader. The purpose of this function is to streamline the creation of PyTorch DataLoaders, allowing for easy
    utilization in a PyTorch training or inference pipeline.

    The PyTorch DataLoader handles the process of drawing batches of data from a dataset, which is essential when
    training models. This function further extends this functionality by implementing data shuffling and utilizing
    multiple worker threads for asynchronous data loading, thereby optimizing the data loading process during model
    training.

    Args:
        dataset: The dataset to be loaded.
        batch_size: The size of each batch the DataLoader will return.
        shuffle: Whether the data should be shuffled before batching.
        num_workers: The number of worker threads to use for data loading.

    Functionality:
        Converts a given dataset into a PyTorch DataLoader.

    :rtype: DataLoader object
    :param dataset: The dataset to be loaded.
    :param batch_size: The size of each batch the DataLoader will return.
    :param shuffle: Whether the data should be shuffled before batching.
    :param num_workers: The number of worker threads to use for data loading.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def log_metrics(writer, log_dict):
    """
    Helper function to log metrics to TensorBoard.

    :param log_dict: Dictionary to log all the metrics to tensorboard. It must contain the keys {epoch,accuracy, loss, lr,}
    :type: log_dict
    :param writer: TensorBoard writer object.
    """

    for key,value in log_dict.items():
        if key not in ['phase','epoch']:
            writer.add_scalar(f'{key}'.capitalize()+f'/{log_dict["phase"]}', value, log_dict["epoch"] + 1)


    # writer.add_scalar(f'Loss/{phase}', loss, epoch + 1)
    # writer.add_scalar(f'Accuracy/{phase}', acc, epoch + 1)
    # for key,value in kwargs:
    #     writer.add_scalar(f'{key}'.capitalize()+f'/{phase}', value, epoch + 1)


    print(f"EPOCH {log_dict['epoch'] + 1:>3}: {log_dict['phase']} accuracy: {log_dict['accuracy']:>3.2f}, "
          f"{log_dict['phase']} Loss: {log_dict['loss']:>9.8f}, LRate {log_dict['lr']:>9.8f} ",
          flush=True)

def log_hparams_metrics(writer,hparam_dict,metric_dict,epoch = 0):
    """
    Helper function to log metrics to TensorBoard. That accepts the logging of hyperparameters too.
    It allows to display the hyperparameters as well in a tensorboard instance. Furthermore it logs everything in just
    one tensorboard log.

    :param writer: Summary Writer Object
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param hparam_dict:
    :type hparam_dict: dict
    :param metric_dict:
    :type metric_dict: dict
    :param epoch: Step on the x-axis to log the results
    :type epoch: int

    """
    exp, ssi, sei = summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete=None)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        if v is not None:
            writer.add_scalar(k, v,global_step=epoch)





def get_PT_Dataset(dataloader):
    """
    Retrieve the underlying dataset from a PyTorch data loader.

    :param dataloader: DataLoader object.
    :return: Dataset object.
    """
    return dataloader.dataset


def get_TF_Dataset(dataloader):
    """
    Retrieve the underlying dataset from a TensorFlow data loader.

    :param dataloader: DatasetWithLen object.
    :return: Dataset object.
    """
    return dataloader.tf_dataset


def get_dataset(dataloader):
    """
    The `get_dataset` function is an interface to extract the underlying dataset from a dataloader, irrespective of
    the deep learning framework being used, i.e., TensorFlow or PyTorch. The versatility of this function makes it
    integral to any pipeline designed to be flexible across both TensorFlow and PyTorch frameworks.

    Given a dataloader object, this function first determines the deep learning framework currently in use by referring
    to the `DL_FRAMEWORK` config parameter variable. If the framework is TensorFlow, it invokes the `get_TF_Dataset`
    function to retrieve the dataset. Alternatively, if PyTorch is being used, the `get_PT_Dataset` function is called.
    This abstracts away the intricacies of handling different deep learning frameworks, thereby simplifying the process
    of working with datasets across TensorFlow and PyTorch.

    Args:
        dataloader: DataLoader from PyTorch or DatasetWithLen from TensorFlow.

    Functionality:
        Extracts the underlying dataset from a dataloader, be it from PyTorch or TensorFlow.

    :rtype: Dataset object
    :param dataloader: DataLoader in case of PyTorch and DatasetWithLen in case of TensorFlow.
    """
    if DL_FRAMEWORK == 'tensorflow':
        return get_TF_Dataset(dataloader)
    else:
        return get_PT_Dataset(dataloader)
