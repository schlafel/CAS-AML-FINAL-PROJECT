import tensorflow as tf

from tf_models import *
from tf_data_models import TF_ASL_Dataset
import os
from config import *
import datetime
import tqdm




def train():
    dm = TF_ASL_Dataset(batch_size=64,
                        path_train=os.path.join(ROOT_PATH,RAW_DATA_DIR,TRAIN_CSV_FILE),
                        MAX_SEQUENCES=MAX_SEQUENCES)
    #Results in a size of (64,150,146)
    input_shape = (MAX_SEQUENCES,146)

    #Hack to get tf_dataset
    def data_generator():
        for i in range(dm.len):
            yield dm.getitem(i)  # edited regaring to @Inigo-13 comment

    tf_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=(
        tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))

    tf_batched = tf_dataset.batch(8)
    for batch, labels in tf_batched.take(1):
        print(batch)
        break


    model = TransformerClassifierModel(
        num_classes=250,
        input_shape=input_shape,
        num_heads=10,
        dff=128,
        num_encoder_blocks=5,
        dropout_rate=0.1,
    )
    model.compile()
    model.build(input_shape=input_shape)

    print(model.summary())



    ## train model
    model.fit_generator(dm)

    print("done")
    for batch in tqdm.tqdm(dm,total=dm.__len__()):
        X,y = batch

        tf.argmax(model(X),axis = 1)
        print("\n", datetime.datetime.now(), X.shape,y.shape)




if __name__ == '__main__':
    MAX_SEQUENCES = 140
    train()

