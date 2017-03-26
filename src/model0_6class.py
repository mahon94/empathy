from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator # for data augmentation if needed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, normalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, adam, RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from log import save_model, save_config, save_result
from keras.models import model_from_json
import numpy as np
import sys

batch_size = 64
nb_classes = 6
nb_epoch = 200

img_rows, img_cols = 48, 48
img_channels = 1

z
#load data
dir = '../data/'
X_fname = dir + 'X_train_Training_fullsplit.npy'
y_fname = dir + 'y_train_Training_fullsplit.npy'
X_train = np.load(X_fname)
y_train = np.load(y_fname)
X_train = X_train.astype('float32')

X_fname = dir + 'X_train_PublicTest_fullsplit.npy'
y_fname = dir + 'y_train_PublicTest_fullsplit.npy'
X_test = np.load(X_fname)
y_test = np.load(y_fname)
X_test = X_test.astype('float32')

# setup info:
print('X_train shape: ', X_train.shape )# (n_sample, 1, 48, 48)
print('y_train shape: ', y_train.shape) # (n_sample, n_categories)
print('  img size: ', X_train.shape[2], X_train.shape[3])
print('batch size: ', batch_size)
print('  nb_epoch: ', nb_epoch)

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',
                input_shape=(1, X_train.shape[2], X_train.shape[3]),
                        activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.25))
#CCP2
model.add(Convolution2D(128, 3, 3, border_mode='same',
                        activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

#CCP3
model.add(Convolution2D(256, 3, 3, border_mode='same',
                        activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

#CCP4
model.add(Convolution2D(512, 3, 3, border_mode='same',
                        activation='relu'))
model.add(Convolution2D(512, 3, 3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

#CCP5
#CCP4
model.add(Convolution2D(512, 3, 3, border_mode='same',
                        activation='relu'))
model.add(Convolution2D(512, 3, 3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print('Not using data augmentation.')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.3,# data=(X_test, y_test),
          shuffle=True,verbose=1)


# print(X_train[:2])
# print(y_train[:2])
# print(X_train.shape)
# print(y_train.shape)
loss_and_metrics1 = model.evaluate(X_train, y_train, batch_size=128, verbose=1)
loss_and_metrics2 = model.evaluate(X_test, y_test, batch_size=128)
print(loss_and_metrics1)
print(loss_and_metrics2)

#save model
notes = 'medium set 100'
save_model(model, model.to_json(), '../data/results/')
save_config(model.get_config(), '../data/results/')
# save_result(loss_and_metrics1, 'train ecaluation, batch size=128', '../data/results/')
# save_result(loss_and_metrics2, 'test ecaluation, batch size=128', '../data/results/')

