from __future__ import print_function
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator # for data augmentation if needed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, normalization
from keras.layers import Convolution2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import SGD, adam, RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import ELU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, BaseLogger
# from log import save_model, save_config, save_result
from keras.models import model_from_json
# from keras.utils import plot_model
import numpy as np
import sys
import time
import os

from shared import *

if len(sys.argv) != 3:
    print('Usage: python visualize.py <checkpoint> <model>')
    exit(0)

# # -------------------------------------------------
# Background config:
weights_path = sys.argv[1]
model_path = sys.argv[2]


model = load_model()