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
import cv2
# from videoEmpathy import load_model
from shared import *
if len(sys.argv) != 4:
    print ('Usage: python train.py <checkpoint> <model> <dataset_dir>')
    exit(0)


# -------------------------------------------------
# Background config:
weights_path = sys.argv[1]
model_path = sys.argv[2]
data_path = sys.argv[3]

model = load_model(model_path, weights_path)
X_train, y_train, X_val, y_val, X_test, y_test, datagen = load_data(data_path)
X_train /= 255.0
X_val /= 255.0
X_test /= 255.0
#------------------------------------------------
# model.compile()

val_acc = model.evaluate(X_val,y_val,batch_size=128, verbose=1)
test_acc = model.evaluate(X_test,y_test,batch_size=128, verbose=1)
print('validation accuracy: ', val_acc)
print('test accuracy: ', test_acc)

for f in os.listdir('../data/sampletest/'):
    myimage = cv2.imread('../data/sampletest/'+f, cv2.IMREAD_GRAYSCALE)
    # myimage = image.img_to_array(myimage)
    print(myimage.shape)
    myimage = np.expand_dims(myimage, axis=0)
    myimage = np.expand_dims(myimage, axis=0)
    print(myimage.shape)
    myimage = myimage.astype('float32')
    myimage = myimage / 255.0
    prediction = model.predict(myimage,batch_size=1)
    pred_str = EMOTIONS[np.argmax(prediction)]
    print(pred_str, f, '\n')

