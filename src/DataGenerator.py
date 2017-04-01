from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
from PIL import Image
import cv2
import sys
from matplotlib import pyplot as plt
import os
# fer2013 dataset:
# Training       28709
# PrivateTest     3589
# PublicTest      3589

# emotion labels from FER2013:
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']


def show(img):
    show_image = img.reshape(48, 48)

    # plt.imshow(show_image, cmap=cm.binary)
    plt.imshow(show_image, cmap='gray')
def reconstruct(pix_str, histequalize, size=(48,48)):
    pix_arr = np.array(map(int, pix_str.split()))
    pix_arr = pix_arr.reshape(size)
    if histequalize:
        pix= pix_arr.astype('uint8')
        pix_image = cv2.cvtColor(pix, cv2.COLOR_GRAY2BGR)
        pix_image = cv2.cvtColor(pix_image, cv2.COLOR_BGR2GRAY)
        pix_equl = cv2.equalizeHist(pix_image)
        # clahe =  cv2.createCLAHE(tileGridSize=(8, 8))
        # pix_equl = clahe.apply(pix_image)
        pix_arr = pix_equl

    return pix_arr

def emotion_count(y_train, classes, verbose=True):
    emo_classcount = {}
    print 'Disgust classified as Angry'
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        if verbose:
            print '{}: {} with {} samples'.format(new_num, _class, class_count)
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount


def load_data(sample_split=0.3, usage='Training', to_cat=True, verbose=True,
              classes=['Angry','Happy'], histequalize=False, normalize=False,
              filepath='../data/fer2013.csv'):

    '''sample a split of data with specified usage and only mentioned classes'''
    df = pd.read_csv(filepath)
    # print df.tail()
    # print df.Usage.value_counts()
    df = df[df.Usage == usage]
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(data.index, int(len(data)*sample_split))
    data = data.ix[rows]
    print '{} set for {}: {}'.format(usage, classes, data.shape)
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x, histequalize))
    print ' CLAHE on data done! (contrast limiting adaptive histogram equalization)'

    x = np.array([mat for mat in data.pixels]) # (n_samples, img_width, img_height)

    if normalize: # do normalization
        # x = x.reshape(x.shape[0],-1) - x.reshape(-1, 48*48).mean(axis=1).reshape(-1, 1)
        # x = x+100
        each_pixel_mean = x.mean(axis=0)
        each_pixle_std = np.std(x, axis=0)
        print "mean" , each_pixel_mean
        print "std" , each_pixle_std



        x = np.divide(np.subtract(x,each_pixel_mean),each_pixle_std)
        each_pixel_mean = x.mean(axis=0)
        each_pixle_std = np.std(x, axis=0)
        print "mean" , each_pixel_mean
        print "std" , each_pixle_std
        x = x.reshape(x.shape[0], 48, -1)
        print x[0][0]

        print 'normalizing data'

    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    y_train, new_dict = emotion_count(data.emotion, classes, verbose)
    print new_dict
    if to_cat:
        y_train = to_categorical(y_train)
    return X_train, y_train, new_dict


def save_data(X_train, y_train, fname='', folder='../data/', save_image = False, usage = ''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_image:
        if not os.path.exists(folder+'images/'+usage ):
            os.makedirs(folder+'images/'+usage )
        for i in range(len(X_train)):
            im = Image.fromarray(X_train[i][0].astype('uint8'))
            im.save(folder+'images/'+usage+'/' + str(y_train[i].argmax()) + '_' + emo[y_train[i].argmax()]
                    + '_' + str(i) + fname + '.png')
    np.save(folder + 'X_' + fname, X_train)
    np.save(folder + 'y_' + fname, y_train)

if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print 'Making moves...'
    emo = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
    usages = ['PrivateTest']#, 'PublicTest', 'Training']
    for usage in usages:
        print usage
        X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                               classes=emo,
                                               usage=usage,
                                               histequalize=False,#True,
                                               normalize=True,#False,#
                                               verbose=True)

        print 'Saving...'


        # print X_train[:2]
        print X_train[0][0]
        print X_train[0][0].shape
        print type(X_train[0][0])

        # save_data(X_train, y_train, fname= usage + '_fullsplit',
        #           folder='../data/alaki/', save_image=False, usage=usage)
        #
        # print X_train.shape
        # print y_train.shape
        # print 'Done!'
