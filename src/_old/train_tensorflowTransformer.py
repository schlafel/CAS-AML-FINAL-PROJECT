import tensorflow as tf
from src._old.tf_models import TransformerClassifierModel
from src._old.tf_data import get_TF_ASL_DATASET
import os
from config import *


def train():
    dm = get_TF_ASL_DATASET(
        max_seq_length=MAX_SEQUENCES,
                        batch_size=512)
    input_shape = (MAX_SEQUENCES,368)
    print(MAX_SEQUENCES)
    for item in dm.take(1):
        print(item[0].shape)
    print("done")

    model = TransformerClassifierModel(
        num_classes=250,
        input_shape=input_shape,
        num_heads=2,
        dff=128,
        num_encoder_blocks=2,
        dropout_rate=0.1,
    )

    model.compile(run_eagerly = True)




    raw_input = (1, *input_shape)
    y = model(tf.ones(shape=(raw_input)))
    print("Ran through model",raw_input,"output shapes",y.shape)
    print(model.summary())
    model.fit(dm,epochs=10,
              #steps_per_epoch=10,
              )


    export_dir = os.path.join(ROOT_PATH,MODEL_DIR,"TRANSFORMER_TF")

    #Save the model
    model.save(export_dir)

    # BRANCH A --> tf.lite.TFLiteConverter.from_saved_model()
    saved_model_converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = saved_model_converter.convert()

    ### BRANCH B --> tf.lite.TFLiteConverter.from_keras_model()
    # keras_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = keras_model_converter.convert()

    ####################################################
    ###   Step 6 – Save Newly Created TFLite Model   ###
    ####################################################

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


    print("Saved Model! :-)")
    # ## train model
    # model.fit_generator(dm)
    #
    # print("done")
    # for batch in tqdm.tqdm(dm,total=dm.__len__()):
    #     X,y = batch
    #
    #     tf.argmax(model(X),axis = 1)
    #     print("\n", datetime.datetime.now(), X.shape,y.shape)




if __name__ == '__main__':

    train()

