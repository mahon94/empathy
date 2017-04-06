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
from scipy.misc import imsave
from keras.models import model_from_json
# from keras.utils import plot_model
import numpy as np
import sys
import time
import os
import math
from shared import *

# IMG_ROWS = IMG_ROWS * 2
# IMG_COLS = IMG_COLS * 2

if len(sys.argv) != 4:
    print('Usage: python visualize.py <checkpoint> <model> <name of model>')
    exit(0)

# # -------------------------------------------------
# Background config:/home/maryam/tensorflow/starter/data/results/kim_7_simple_data_aug_run2
weights_path = sys.argv[1] #'../data/results/kim_7_simple_data_aug_run2/weights.81-1.06-0.62385.hdf5'# sys.argv[1]
model_path = sys.argv[2] #'../data/results/kim_7_simple_data_aug_run2/Sun Apr  2 00:26:13 2017.json' #sys.argv[2]
name = sys.argv[3] #'larger_images'

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # if K.image_data_format() == 'channels_first':
    x = x.transpose((1, 2, 0))


    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)


model = load_model(model_path, weights_path)

print(model.summary())
input_img = model.input

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


layer_dict = dict([(layer.name, layer) for layer in model.layers])

print(layer_dict)
layer_name = 'convolution2d_6'#'convolution2d_3'

kept_filters = []
for filter_index in range(0, layer_dict[layer_name].output.shape[1]):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    #----- ta inja fahmidam

    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    # if K.image_data_format() == 'channels_first':
    input_img_data = np.random.random((1, 1, IMG_ROWS, IMG_COLS))
    # else:
    #     input_img_data = np.random.random((1, IMG_ROWS, IMG_COLS, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    iteration_num=50
    for i in range(iteration_num):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = int(min([16, math.sqrt(len(kept_filters))]))
print('n:',n)
# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * IMG_ROWS + (n - 1) * margin
height = n * IMG_COLS + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

print(len(kept_filters))

# fill the picture with our saved filters

for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(IMG_ROWS + margin) * i: (IMG_ROWS + margin) * i + IMG_ROWS,
                         (IMG_COLS + margin) * j: (IMG_COLS + margin) * j + IMG_COLS, :] = img

# save the result to disk
imsave('../LayerVisualizations/%s_%s_stitched_filters_%dx%d_%d.png' % (name, layer_name, n, n, iteration_num), stitched_filters)



