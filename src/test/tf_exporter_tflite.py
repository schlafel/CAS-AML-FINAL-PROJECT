import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import os

class My_ConvBlock(layers.Layer):
    def __init__(self,conv_blocks = [32,64,128],kernel_size = 3, dropout = 0.2):
        super(My_ConvBlock, self).__init__()
        self.ConvBlock1 = layers.Conv2D(conv_blocks[0],kernel_size = kernel_size)
        self.ConvBlock2 = layers.Conv2D(conv_blocks[1],kernel_size = kernel_size)
        self.ConvBlock3 = layers.Conv2D(conv_blocks[2],kernel_size = kernel_size)
        self.dropout = layers.Dropout(dropout)

    def call(self,inputs, training = False,**kwargs):
        x = self.ConvBlock1(inputs)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        return x


class My_DenseLayer(layers.Layer):
    def __init__(self,n_units,dropout = 0.1):
        super(My_DenseLayer, self).__init__()


        self.fc = [layers.Dense(n_units) for n in range(5)]
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self,x, training = False, **kwargs):

        for fc in self.fc:
            x = fc(x)
        x = self.bn(x,training=training)
        x = self.dropout(x)
        return x


class MySimpleModel(Model):
    def __init__(self,n_classes = 50):
        super(MySimpleModel, self).__init__()
        self.conv_Block = My_ConvBlock(kernel_size = 3)
        self.flat = layers.Flatten()
        self.fc_layer = My_DenseLayer(n_units=50)
        self.dense = layers.Dense(n_classes)

    def call(self,inputs, training = False,**kwargs):
        x = self.conv_Block(inputs)
        x = self.flat(x)

        x = self.fc_layer(x,training=training)
        x = tf.nn.relu(self.dense(x))

        return tf.nn.softmax(x)





if __name__ == '__main__':
    print(tf.__version__)
    model = MySimpleModel(n_classes=10)
    model.compile(
     loss = 'categorical_crossentropy',
     metrics = ['accuracy'],
     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))

    raw_input = (1, 50,50,1)
    y = model(tf.ones(shape=(raw_input)))
    print(model.summary())
    print("done", y.shape)

    print("weights:", len(model.weights))
    print("trainable weights:", len(model.trainable_weights))



    export_dir = os.path.join(".","TRANSFORMER_TF")

    #Save the model
    model.save(export_dir)

    # BRANCH A --> tf.lite.TFLiteConverter.from_saved_model()
    saved_model_converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = saved_model_converter.convert()



    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


    print("Saved Model! :-)")

