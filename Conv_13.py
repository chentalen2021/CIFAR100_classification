#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, Input, Model, regularizers
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Activation,BatchNormalization, MaxPooling2D, Flatten
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
# import centre_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["tensorflow.keras_BACKEND"] = "plaidml.tensorflow.keras.backend"

#Allow the GPU memory growth for deep learning
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices

#%%
#Build a 13-layer CNN
def Conv13_model():
    weight_decay = 4e-4

        #kernel +, whereas h * d -
    #Input layer -> 32 * 32 *3
    #Build a 13-layer CNN
        #kernel +, whereas h * d -
    #Input layer -> 32 * 32 *3
    inputs = keras.Input(shape=(32, 32, 3))
    #1st and 2nd Conv -> Conv 64 *2 + max_pooling
    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

    #3rd and 4th Conv -> Conv 128 *2 + max_pooling
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

    #5th and 6th Conv -> Conv 256 *2 + max_pooling
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

    #7th, 8th, 9th, and 10th Conv -> Conv 512 *4 + max_pooling *2
    #Finally get a 1*1*512(channel) Tensor
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)
    x = layers.Flatten()(x)

    #11th and 12th FC layer
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)

    #Output layer -> 100
    output = layers.Dense(100)(x)
    output

    cnn_13 = Model(inputs=inputs,outputs=output)

    return cnn_13
# %%
if __name__=="__main__":
    pass
