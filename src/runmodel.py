from __future__ import print_function
import keras.backend as K
from keras.backend.common import set_image_data_format
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

import itertools
# from videoEmpathy import load_model
from sklearn.metrics import confusion_matrix
import matplotlib
from matplotlib import pyplot as plt

from shared import *
NB_CLASSES = 7

set_image_data_format('channels_first')
np.set_printoptions(precision=2)

#
# def plot_confusion_matrix(y_true, y_pred,normalized = False, cmap=plt.cm.Blues):
#     cm = confusion_matrix(y_true, y_pred)
#     print(cm)
#     # if normalized:
#     #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     #     print("Normalized confusion matrix")
#     # thresh = cm.max() / 2.
#     # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     #     plt.text(j, i, cm[i, j],
#     #              horizontalalignment="center",
#     #              color="white" if cm[i, j] > thresh else "black")
#     fig = plt.figure(figsize=(NB_CLASSES,NB_CLASSES))
#     matplotlib.rcParams.update({'font.size': 16})
#     ax  = fig.add_subplot(111)
#     matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     fig.colorbar(matrix)
#     for i in range(0,NB_CLASSES):
#         for j in range(0,NB_CLASSES):
#             ax.text(j,i,cm[i,j],va='center', ha='center')
#     # ax.set_title('Confusion Matrix')
#     ticks = np.arange(len(EMOTIONS))
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(EMOTIONS, rotation=45)
#     ax.set_yticks(ticks)
#     ax.set_yticklabels(EMOTIONS)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()


# if len(sys.argv) != 4:
#     print ('Usage: python train.py <checkpoint> <model> <dataset_dir>')
#     exit(0)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    cm = np.round(cm,2)

    thresh = cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 700 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# -------------------------------------------------
# Background config:
weights_path = sys.argv[1]#'../data/results/kim_7_simple_data_aug_run2/weights.114-1.06-0.61577.hdf5'#VGGnet_7_simple_data_aug/weights.145-0.92-0.67707.hdf5' # sys.argv[1]
model_path = sys.argv[2]#'../data/results/kim_7_simple_data_aug_run2/Sun Apr  2 00:26:13 2017.json'#VGGnet_7_simple_data_aug/Mon Apr  3 01:51:01 2017.json' # sys.argv[2]
data_path = sys.argv[3]#'../data/7class_simple_normalized/'#sys.argv[3]



model = load_model(model_path, weights_path)
X_train, y_train, X_val, y_val, X_test, y_test, datagen = load_data(data_path)
# X_train /= 255.0
# X_val /= 255.0
# X_test /= 255.0
#------------------------------------------------
# model.compile()

#
# val_acc = model.evaluate(X_val,y_val,batch_size=128, verbose=1)
# test_acc = model.evaluate(X_test,y_test,batch_size=128, verbose=1)
# print('validation accuracy: ', val_acc)
# print('test accuracy: ', test_acc)


y_prob = model.predict(X_test,batch_size=32)
y_pred = [np.argmax(p) for p in y_prob]
y_true = [np.argmax(p) for p in y_test]


print(y_prob[0])
print(y_pred[0])
print(y_true[0])
print(y_pred[:50])
print(y_true[:50])
print(type(y_true))
print(y_true.index(1))
cm = confusion_matrix(y_true,y_pred)
print(cm)

plt.figure()
# plot_confusion_matrix(y_true, y_pred, normalized=True, cmap=plt.cm.YlGnBu )
plot_confusion_matrix(cm, EMOTIONS, normalize=False, cmap=plt.cm.YlGnBu )

# for f in os.listdir('../data/sampletest/'):
#     myimage = cv2.imread('../data/sampletest/'+f, cv2.IMREAD_GRAYSCALE)
#     # myimage = image.img_to_array(myimage)
#     print(myimage.shape)
#     myimage = np.expand_dims(myimage, axis=0)
#     myimage = np.expand_dims(myimage, axis=0)
#     print(myimage.shape)
#     myimage = myimage.astype('float32')
#     myimage = myimage / 255.0
#     prediction = model.predict(myimage,batch_size=1)
#     pred_str = EMOTIONS[np.argmax(prediction)]
#     print(pred_str, f, '\n')
#
