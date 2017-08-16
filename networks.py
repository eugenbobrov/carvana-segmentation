#!/usr/bin/env python3
from keras.models import Sequential, Model
from keras.layers import (Input, InputLayer, Conv2D, MaxPool2D, UpSampling2D,
                          Conv2DTranspose, Dropout, concatenate)
from parameters import height, width


def segnet():
    model = Sequential()
    encoding_layers = [
        InputLayer((height, width, 1)),
        Conv2D(32, 3, activation='relu', padding='same'),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPool2D((2, 2)),

        Conv2D(64, 3, activation='relu', padding='same'),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPool2D((2, 2)),

        Conv2D(128, 3, activation='relu', padding='same'),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPool2D((2, 2)),

        Conv2D(256, 3, activation='relu', padding='same'),
        Conv2D(256, 3, activation='relu', padding='same'),
        MaxPool2D((2, 2)),

        Conv2D(512, 3, activation='relu', padding='same'),
        Conv2D(512, 3, activation='relu', padding='same'),
    ]
    decoding_layers = [
        Conv2D(512, 3, activation='relu', padding='same'),
        Conv2D(512, 3, activation='relu', padding='same'),

        UpSampling2D((2, 2)),
        Conv2D(256, 3, activation='relu', padding='same'),
        Conv2D(256, 3, activation='relu', padding='same'),

        UpSampling2D((2, 2)),
        Conv2D(128, 3, activation='relu', padding='same'),
        Conv2D(128, 3, activation='relu', padding='same'),

        UpSampling2D((2, 2)),
        Conv2D(64, 3, activation='relu', padding='same'),
        Conv2D(64, 3, activation='relu', padding='same'),

        UpSampling2D((2, 2)),
        Conv2D(32, 3, activation='relu', padding='same'),
        Conv2D(32, 3, activation='relu', padding='same'),

        Conv2D(1, 1, activation='sigmoid', padding='valid')
    ]
    [model.add(l) for l in encoding_layers]
    [model.add(l) for l in decoding_layers]
    return model


def unet():
    inputs = Input((height, width, 1))
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D((2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
    cont5 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv5)

    upsm6 = concatenate([cont5, conv4])
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(upsm6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
    cont6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)

    upsm7 = concatenate([cont6, conv3])
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(upsm7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)
    cont7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)

    upsm8 = concatenate([cont7, conv2])
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(upsm8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
    cont8 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)

    upsm9 = concatenate([cont8, conv1])
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(upsm9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid', padding='valid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    return model