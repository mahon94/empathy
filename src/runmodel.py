from __future__ import print_function
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator # for data augmentation if needed
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, normalization
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

if len(sys.argv) != 3:
    print ('Usage: python train.py <checkpoint> <model> <dataset_dir>')
    exit(0)


# -------------------------------------------------
# Background config:
weights_path = sys.argv[1]
model_path = sys.argv[2]
data_path = sys.argv[3]

# -------------------------------------------------
# load json and create model

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# -------------------------------------------------
# load weights into new model

model.load_weights(weights_path)
print("Loaded model from disk")

#------------------------------------------------
model.compile()

