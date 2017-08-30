#!/usr/bin/env python3
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPool2D, Conv2DTranspose,
                          BatchNormalization, concatenate)


def down2x(kernels, x):
    l = MaxPool2D((2, 2))(x)
    l = Conv2D(kernels, 5, activation='relu', padding='same')(l)
    l = Conv2D(kernels, 5, activation='relu', padding='same')(l)
    return BatchNormalization()(l)


def up2x(kernels, x, y):
    l = Conv2DTranspose(kernels, 4, strides=(2, 2), padding='same')(x)
    l = concatenate([l, y])
    l = Conv2D(kernels, 5, activation='relu', padding='same')(l)
    l = Conv2D(kernels, 5, activation='relu', padding='same')(l)
    return BatchNormalization()(l)


def unet():
    inputs = Input((None, None, 3))

    l1 = BatchNormalization()(inputs)
    l1 = Conv2D(16, 5, activation='relu', padding='same')(l1)
    l1 = Conv2D(16, 5, activation='relu', padding='same')(l1)
    l1 = BatchNormalization()(l1)

    l2 = down2x(32, l1)
    l3 = down2x(64, l2)
    l4 = down2x(128, l3)

    l5 = down2x(256, l4)

    l6 = up2x(128, l5, l4)
    l7 = up2x(64, l6, l3)
    l8 = up2x(32, l7, l2)

    l9 = up2x(16, l8, l1)
    outputs = Conv2D(1, 1, activation='sigmoid')(l9)
    return Model([inputs], [outputs])