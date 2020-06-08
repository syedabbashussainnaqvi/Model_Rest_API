import keras
import random
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from skimage.io import imread
from keras import backend as K
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
import tifffile as tiff

def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

def predictS1(path):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(120,120,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=512,activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=512,activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=19, activation="sigmoid"))
    model.compile(optimizers.adam(), loss="binary_crossentropy", metrics=['accuracy', recall, precision, f1])
    model.load_weights('ben_data_vgg_s1.h5')
    label = ["Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops","Pastures", "Complex cultivation patterns",
         "Land principally occupied by agriculture, with significant areas of natural vegetation",
          "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
          "Natural grassland and sparsely vegetated areas", "Moors, heathland and sclerophyllous vegetation",
          "Transitional woodland, shrub", "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands",
          "Inland waters", "Marine waters"]
#     10: means 10 onwards here its 11 and 12 band
    image = np.array(tiff.imread(path), dtype=float)[:,:,10:]
    img = np.reshape(image, (1, 120, 120, 2))
    r=model.predict(img)
    r[r>=.5]=1
    r[r<.5]=0
    l=np.where(r==1)[1]
    labels=[]
    [labels.append(label[i]) for i in l]
    return labels
