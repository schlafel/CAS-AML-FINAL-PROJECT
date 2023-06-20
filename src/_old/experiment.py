from src._old.preprocessing_layers import *
from src.data.data_utils import load_relevant_data_subset
import pandas as pd
import os

if __name__ == '__main__':
    prep_layer = PreprocessLayer(
        useful_landmarks= USEFUL_ALL_LANDMARKS,
                                 hand_idxs= USEFUL_HAND_LANDMARKS
    )
    COLUMNS_TO_USE = ["x","y","z"]


    # Load the training data
    df_train = pd.read_csv(os.path.join(ROOT_PATH, RAW_DATA_DIR, TRAIN_CSV_FILE))

    sample = df_train.sample(1)
    sample_file = sample.path
    print(sample_file)

    arr_relevant = load_relevant_data_subset(os.path.join(ROOT_PATH,RAW_DATA_DIR,sample_file.values[0]),
                                             cols_to_use=COLUMNS_TO_USE)
    print(arr_relevant.shape,arr_relevant.dtype)

    #Get number of empty frames
    idx_nan = np.where(
        np.nansum(arr_relevant[:, USEFUL_HAND_LANDMARKS, 0:2],
                  axis=(1, 2)) == 0)[0]
    idx_nan_tf = tf.squeeze(tf.where(tf.experimental.numpy.nansum(arr_relevant[:, USEFUL_HAND_LANDMARKS, 0:2], axis=(1, 2)) == 0))

    print(f"Found {len(idx_nan)} values out of {arr_relevant.shape[0]}" )


    arr_prep,idx = prep_layer(arr_relevant)
    print(arr_prep.shape,idx.shape)

    print(idx.numpy())

    print(tf.squeeze(tf.where(tf.experimental.numpy.nansum(arr_prep[:, :, 0:2], axis=(1, 2)) == 0))
)

    #print(arr_prep)