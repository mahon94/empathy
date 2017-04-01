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
import numpy as np
import sys
import time
import os

from shared import *

#params
BATCH_SIZE = 128
NB_CLASSES = 6
NB_EPOCH = 500
CONTINUEING = False #True
PRINT_MODEL_SUMMARY = True
SAVE_MODEL_PLOT = True


#-------------------------------------------------------
# get pathes
if len(sys.argv) <= 3:
    print('Usage: python model0_6class.py <data_path> <network_name_to Store>'
          '<data-augmentation> <loading_weights> <weights address>')
    exit(0)

print(len(sys.argv))

data_path = sys.argv[1]#'../data/' #
network_name = sys.argv[2] # 'Zhang_simple_data' #'getSthToWork' #
DATA_AUGMENTATION = (sys.argv[3] == 'True')
if len(sys.argv) > 4:
    CONTINUEING = (sys.argv[4] =='True')
    weight_path = sys.argv[5]

store_path = '../data/results/' + network_name + '/'
#make path for savin weights and model.json
if not os.path.exists(store_path):
    os.makedirs(store_path)
if not os.path.exists(store_path+'log/'):
    os.makedirs(store_path+'log/')
#-------------------------------------------------------

def compile_model(model):
    myadam = adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=myadam,  # 'adam',
                  metrics=['accuracy'])
    return model

myELU = ELU()

def size(_model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in _model.trainable_weights])


def load_model(path):
    model.load_weights(path)

def describe():
    # setup info:
    print('X_train shape: ', X_train.shape)  # (n_sample, 1, 48, 48)
    print('y_train shape: ', y_train.shape)  # (n_sample, n_categories)
    print('  img size: ', X_train.shape[2], X_train.shape[3])
    print('batch size: ', BATCH_SIZE)
    print('  nb_epoch: ', NB_EPOCH)



def zhangnet(network_name = 'network14_zhang_equalized_data'):
    model = Sequential()

    #CPN1
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                    input_shape=(1, X_train.shape[2], X_train.shape[3]),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    # model.add(Dropout(0.25))
    #CPN2
    model.add(Convolution2D(96, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    # model.add(Dropout(0.25))
    #CPN3
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    # model.add(Dropout(0.25))
    #CN4
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(normalization.BatchNormalization())
    # model.add(Dropout(0.25))
    #F 2048
    model.add(Flatten())
    model.add(Dense(1024))#, activity_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    return model

def test2(network_name = 'network14_zhang_equalized_data'):
    # model architecture:
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            input_shape=(1, X_train.shape[2], X_train.shape[3])))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model

def kim():
    # model architecture:
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu',
                            input_shape=(1, X_train.shape[2], X_train.shape[3])))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(normalization.BatchNormalization())


    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(1024, activity_regularizer=regularizers.l2(0.0001)))
    model.add(myELU)
    model.add(Dropout(0.5))
    model.add(normalization.BatchNormalization())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model

def shallow():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            input_shape=(1, X_train.shape[2], X_train.shape[3])))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    # CPN2
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering='th'))
    model.add(normalization.BatchNormalization())

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(512, activity_regularizer=regularizers.l2(0.0001)))
    model.add(myELU)
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def vggnet():
    model = Sequential()

    #CPN1
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                    input_shape=(1, X_train.shape[2], X_train.shape[3]),
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.25))
    #CPN2
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.25))
    #CPN3
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.25))
    #CN4
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering='th'))
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.25))
    #F 2048
    model.add(Flatten())
    model.add(Dense(1024, activity_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    return model

    pass




starttime = time.asctime(time.localtime(time.time()))

X_train, y_train, X_val, y_val, X_test, y_test, datagen = load_data(data_path)
#---------------------------------------------------------
#make model
print("Loading network/training configuration...")
# model = zhangnet(network_name)
# model = test2(network_name)
# model = vggnet()
# model = kim()
model = shallow()
# ---------------------------------------------------------
#callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=25, min_lr=0.0001)
checkpointer = ModelCheckpoint(filepath=store_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.5f}.hdf5', verbose=1,
                               save_best_only=True, mode=max)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=50, verbose=1)

callbks = [reduce_lr, checkpointer, earlystop, TensorBoard(log_dir=store_path+'log/'), BaseLogger()]


if CONTINUEING:
    model.load_weights(weight_path)
    print('loaded previous model')

# Let's train the model using adam
model = compile_model(model)

if CONTINUEING:
    loss_and_metrics2 = model.evaluate(X_val, y_val, batch_size=128, verbose=1)
    loss_and_metrics3 = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
    print('val_acc: ', loss_and_metrics2)
    print('test_acc: ' , loss_and_metrics3)

if PRINT_MODEL_SUMMARY:
    print("Model summary...")
    print(model.summary())

describe()
# save model
save_model(model, store_path)
save_config(model.get_config(), store_path)


if DATA_AUGMENTATION == False:
    hist = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     nb_epoch=NB_EPOCH,
                     validation_data=(X_val, y_val),
                     shuffle=True, verbose=1, callbacks=callbks)

else:

    hist = model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=BATCH_SIZE),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=NB_EPOCH,
                        validation_data=(X_test, y_test),
                        verbose=1, callbacks=callbks)


if SAVE_MODEL_PLOT:
    print("Saving model plot...")
    from keras.utils.visualize_util import plot
    plot(model, to_file=(store_path + 'deepRL.png'), show_shapes=True)

train_val_accuracy = hist.history
train_acc = train_val_accuracy['acc']
val_acc = train_val_accuracy['val_acc']
print('          Done!')
print('     Train acc: ', train_acc[-1])
print('Validation acc: ', val_acc[-1])

loss_and_metrics1 = model.evaluate(X_train, y_train, batch_size=128, verbose=1)
loss_and_metrics2 = model.evaluate(X_val, y_val, batch_size=128, verbose=1)
loss_and_metrics3 = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print(loss_and_metrics1)
print(loss_and_metrics2)
print(loss_and_metrics3)
print(hist.history)

save_result(starttime, BATCH_SIZE, NB_EPOCH, model, size(model),
            loss_and_metrics1[1], loss_and_metrics2[1], loss_and_metrics3[1], history=hist.history,
            dirpath='../data/results/' + network_name + '/')
