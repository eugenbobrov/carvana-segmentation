import os
import numpy as np
import numpy.random as npr
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPool2D, Deconv2D
from keras.losses import binary_crossentropy
from keras import backend as K
from skimage.transform import downscale_local_mean
from skimage.io import imread_collection
from parameters import (height, width, path_in, path_out,
                        train_size, batch_size, scale)


def dice_coef(T, P):
    return (2*K.sum(T*P) + 1) / (K.sum(T) + K.sum(P) + 1)


def bce_dice_loss(T, P):
    return binary_crossentropy(T, P)/2 - dice_coef(T, P)


def get_unet():
    inputs = Input((height, width, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Deconv2D(256, (2, 2), strides=(2, 2),
                                padding='same')(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Deconv2D(128, (2, 2), strides=(2, 2),
                                padding='same')(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Deconv2D(64, (2, 2), strides=(2, 2),
                                padding='same')(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Deconv2D(32, (2, 2), strides=(2, 2),
                                padding='same')(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile('adam', bce_dice_loss, ['accuracy', dice_coef])
    return model


def data_generator():
    train_names = np.array(sorted(os.listdir(path_in + 'train')))
    masks_names = np.array(sorted(os.listdir(path_in + 'train_masks')))
    while True:
        idxs = npr.choice(train_size, batch_size)
        paths = [path_in + 'train/' + s for s in train_names[idxs]]
        train = imread_collection(paths).concatenate()
        paths = [path_in + 'train_masks/' + s for s in masks_names[idxs]]
        masks = imread_collection(paths).concatenate()

        train = downscale_local_mean(train, (1, scale, scale, 3))
        train -= train.mean()
        train /= train.std()

        masks = downscale_local_mean(masks, (1, scale, scale))/255
        masks = masks[:, :, :, None]
        yield train, masks


def optimize_network():
    model = get_unet()
    history = model.fit_generator(
        data_generator(), train_size//batch_size, epochs=5)
    np.save(path_out + 'dice.npy', history.history['dice_coef'])
    np.save(path_out + 'loss.npy', history.history['loss'])
    np.save(path_out + 'accuracy.npy', history.history['acc'])
    model.save_weights(path_out +'unet.h5')


def predict_masks(name, model):
    paths = path_in + 'test/' + name + '_'
    paths = [paths + str(j + 1).zfill(2) + '.jpg' for j in range(16)]
    test = imread_collection(paths).concatenate()
    test = downscale_local_mean(test, (1, scale, scale, 3))
    test -= test.mean()
    test /= test.std()

    masks = model.predict(test, batch_size=8, verbose=1) > 0.5
    return masks[:, :, :, 0]


if __name__ == '__main__':
    optimize_network()