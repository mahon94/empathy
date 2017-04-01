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
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from log import save_model, save_config, save_result
from keras.models import model_from_json
import numpy as np
import sys
import time
import os

#parameters:
IMG_ROWS, IMG_COLS = 48, 48
IMG_CHANNELS = 1
DATA_AUGMENTATION = True #False #
CASC_PATH = '../haarcascades/haarcascade_frontalface_default.xml'
EMOTIONS = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral', 'disgust']

endtime = time.asctime(time.localtime(time.time()))


def load_data(data_path):
    '''loads training, validation and test data and labels'''
    print('loading data from '+ data_path)
    # load data
    # data_path= '../data/'  # histequalized/'
    X_fname = data_path + 'X_Training_fullsplit.npy'
    y_fname = data_path + 'y_Training_fullsplit.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    X_train = X_train.astype('float32')

    X_fname = data_path + 'X_PublicTest_fullsplit.npy'
    y_fname = data_path + 'y_PublicTest_fullsplit.npy'
    X_val = np.load(X_fname)
    y_val = np.load(y_fname)
    X_val = X_val.astype('float32')

    X_fname = data_path + 'X_PrivateTest_fullsplit.npy'
    y_fname = data_path + 'y_PrivateTest_fullsplit.npy'
    X_test = np.load(X_fname)
    y_test = np.load(y_fname)
    X_test = X_test.astype('float32')

    # X_train /= 255.0
    # X_val /= 255.0
    # X_test /= 255.0
    datagen = None

    if DATA_AUGMENTATION:
        print("Using real-time data augmentation")
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True)  # randomly flip images
        datagen.fit(X_train)
    else:
        print('Not using data augmentation.')
    return X_train, y_train, X_val, y_val, X_test, y_test, datagen


def load_model(model_path, weights_path):
    '''load json and create model'''
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # -------------------------------------------------
    # load weights into new model
    model.load_weights(weights_path)
    myadam = adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=myadam,  # 'adam',
                  metrics=['accuracy'])
    # model = compile_model(model)

    print("Loaded model from disk")

    return model


def save_model(model, dirpath='../data/results/'):
    with open(dirpath + endtime +'.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(dirpath + endtime + ".h5")


def save_config(config, dirpath='../data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(endtime + '\n')
        f.write(str(config) + '\n')


def save_result(starttime, batch_size, nb_epoch, model, modelParams, train_acc, val_acc, test_acc,
                history = '', dirpath='../data/results/'):
    with open(dirpath + endtime +'_result_log.txt', 'w') as f:
            f.write(starttime + '_' + endtime + '\n')
            f.write('      batch size: ' + str(batch_size) + ', epoches: ' + str(nb_epoch) + '\n')
            # f.write('         summary: ' + str(modelSummary) + '\n')
            f.write('number of params: ' + str(modelParams) + '\n')
            f.write('       train acc: ' + str(train_acc) + '\n')
            f.write('  validation acc: ' + str(val_acc) + '\n')
            f.write('        test acc: ' + str(test_acc) + '\n')
            f.write('         history: ' + str(history) + '\n')
            orig_stdout = sys.stdout
            sys.stdout = f
            print(model.summary())
            sys.stdout = orig_stdout


