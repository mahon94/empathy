from __future__ import print_function
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator # for data augmentation if needed
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, Embedding
from keras.layers import Convolution2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import SGD, adam, RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from log import save_model, save_config, save_result
from keras.models import model_from_json
import numpy as np
import sys
import time
import os
import cv2
# from videoEmpathy import load_model
from sklearn.metrics import confusion_matrix
import matplotlib
from matplotlib import pyplot as plt

from shared import *

NB_CLASSES = 7
data_path = '../data/7class_normalized_correct/'
X_train, y_train, X_val, y_val, X_test, y_test, datagen = load_data(data_path)

dictionary_size = 3589#300 #should be 256 I think
embeddings_size = 20
num_inputs = 3


model = Sequential()
for i in range(num_inputs):
    model.add(Embedding(input_dim=dictionary_size,output_dim=embeddings_size, input_length=48*48,
                        name='embeddings_{}'.format(i)))
    model.add(Flatten())

model.add(Dense(256, activation='relu', name= 'hidden1'))

model.add(Dense(7, activation='softmax', name='output'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',  # 'adam',
                  metrics=['accuracy'])

embeddings_to_monitor = ['embeddings_{}'.format(i)
                         for i in range(num_inputs)]

metadata_file_name = 'metadata.tsv'
embeddings_metadata = {layer_name: metadata_file_name
                       for layer_name in embeddings_to_monitor}

tb_callback = TensorBoard(histogram_freq=10, write_graph=False,
                          embeddings_freq=100,                          # Store each 100 epochs...
                          embeddings_layer_names=embeddings_to_monitor, # this list of embedding layers...
                          embeddings_metadata=embeddings_metadata)      # with this metadata associated with them.
print(X_test.shape)

X_test = X_test.reshape(-1,48*48)
print(X_test.shape)
model.fit(X_test, y_test, nb_epoch = 1000, batch_size= 512, callbacks=[tb_callback], verbose=1)

