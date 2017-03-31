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
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from log import save_model, save_config, save_result
from keras.models import model_from_json
import numpy as np
import time
import os
# from model0_6class import compile_model
from keras.preprocessing import image
from cv2 import cv
import cv2
import sys
from shared import *


# print(len(sys.argv))
# print(sys.argv[1])



if len(sys.argv) != 3:
    print('Usage: python videoEmpathy.py <checkpoint> <model>')
    exit(0)

# # -------------------------------------------------
# Background config:
weights_path = sys.argv[1]
model_path = sys.argv[2]
#
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
#

def load_model():
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


def brighten(data, b):
    datab = data * b
    return datab


def format_image(myimage):
    # myimage = cv2.imread('../data/images/2_Happy_1306_train.png', cv2.IMREAD_GRAYSCALE)
    # # myimage = image.img_to_array(myimage)
    # print(myimage.shape)
    # myimage = np.expand_dims(myimage, axis=0)
    # myimage = np.expand_dims(myimage, axis=0)
    # print(myimage.shape)
    # myimage = myimage.astype('float32')
    # myimage = myimage / 255.0
    # return myimage
    if len(myimage.shape) > 2 and myimage.shape[2] == 3:
        myimage = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    else:
        myimage = cv2.imdecode(myimage, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        myimage,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    myimage = myimage[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        myimage = cv2.resize(myimage, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    # cv2.imshow("Lol", myimage)
    # cv2.waitKey(0)

    print(myimage.shape)
    myimage = np.expand_dims(myimage, axis=0)
    myimage = np.expand_dims(myimage, axis=0)
    print(myimage.shape)
    myimage = myimage.astype('float32')
    myimage = myimage / 255.0

    return myimage


# Load Model
network = load_model()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('../emojis/' + emotion + '.png', -1))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    if frame is not None:
        tmp = format_image(frame)
        if tmp == None:
            continue
        result = network.predict(tmp, batch_size=1)

        # Draw face in frame
        # for (x,y,w,h) in faces:
        #   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Write results in frame
        if result is not None:
            print(result)
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)

            face_image = feelings_faces[np.argmax(result)]

            # Ugly transparent fix
            for c in range(0, 3):
                frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130,
                                                                                                  c] * (
                                                                                                  1.0 - face_image[:, :,
                                                                                                        3] / 255.0)

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    if cv2.waitKey(1) == 27:
        break  # esc to quit
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
