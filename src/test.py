# Random Flips
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np
from matplotlib import pyplot
# K.set_image_dim_ordering('th')
# load data
dir = '../data/histequalized/'
X_fname = dir + 'X_Training_fullsplit.npy'
y_fname = dir + 'y_Training_fullsplit.npy'
X_train = np.load(X_fname)
y_train = np.load(y_fname)
X_train = X_train.astype('float32')

X_fname = dir + 'X_PublicTest_fullsplit.npy'
y_fname = dir + 'y_PublicTest_fullsplit.npy'
X_val = np.load(X_fname)
y_val = np.load(y_fname)
X_val = X_val.astype('float32')
#
# # define data preparation
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True)  # randomly flip images
datagen.fit(X_train)

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='./test'):
    # create a grid of 3x3 images
    # for i in range(0, 9):
    #     pyplot.subplot(330 + 1 + i)
    #     pyplot.imshow(X_batch[i], cmap=pyplot.get_cmap('gray'))
    # # show the plot
    # pyplot.show()
    break