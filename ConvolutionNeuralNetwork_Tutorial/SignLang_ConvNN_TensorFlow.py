#!/usr/bin/env python

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


def create_model(InputShape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    Args:
    InputShape : Shape of image

    returns Model:Tf keras model
    """

    input_sh = tf.keras.Input(shape = InputShape)
    X = tf.keras.layers.Conv2D(8, (4, 4), strides = 1,padding = 'same', activation='relu')(input_sh)
    P1 =  tf.keras.layers.MaxPool2D((8,8), strides = 8, padding = 'same')(X)
    Z2 = tf.keras.layers.Conv2D(16, (2,2),strides = 1, padding = 'same', activation='relu')(P1)
    P2 = tf.keras.layers.MaxPool2D((4,4), strides = 4, padding = 'same')(Z2)
    F = tf.keras.layers.Flatten()(P2)
    Z3 = tf.keras.layers.Dense(units = 6, activation = 'softmax')(F)
    model = tf.keras.Model(inputs=input_sh, outputs=Z3)

    return model

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
index = 32

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
conv_layers = {}

conv_model = create_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)

df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
