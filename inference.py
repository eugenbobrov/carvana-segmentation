#!/usr/bin/env python3
import os
import numpy as np
import numpy.random as npr
from skimage.transform import downscale_local_mean
from skimage.io import imread_collection
from keras.losses import binary_crossentropy
from keras import backend as K
from networks import segnet, unet
from parameters import (path_in, path_out, scale,
                        train_size, batch_size, valid_size)


def dice_coef(T, P):
    return (2*K.sum(T*P) + 1) / (K.sum(T) + K.sum(P) + 1)


def bce_dice_loss(T, P):
    return binary_crossentropy(T, P)/2 - dice_coef(T, P) + 1


def data_generator(dir_name, data_size):
    data_path = path_in + dir_name +'/'
    masks_path = path_in + dir_name + '_masks/'
    data_names = np.array(sorted(os.listdir(data_path)))
    masks_names = np.array(sorted(os.listdir(masks_path)))
    while True:
        idxs = npr.choice(data_size, batch_size)
        paths = [data_path + s for s in data_names[idxs]]
        data = imread_collection(paths).concatenate()
        paths = [masks_path + s for s in masks_names[idxs]]
        masks = imread_collection(paths).concatenate()

        data = downscale_local_mean(data, (1, scale, scale, 3))
        data -= data.mean()
        data /= data.std()

        masks = downscale_local_mean(masks, (1, scale, scale))/255
        masks = masks[:, :, :, None]
        yield data, masks


def network_learning(valid=True):
    model = unet()
    model.compile('adam', bce_dice_loss, ['accuracy', dice_coef])
    if valid:
        history = model.fit_generator(epochs=5,
            generator=data_generator('train', train_size),
            steps_per_epoch=train_size//batch_size,
            validation_data=data_generator('valid', valid_size),
            validation_steps=valid_size//batch_size)
    else:
        history = model.fit_generator(epochs=5,
            generator=data_generator('train', train_size),
            steps_per_epoch=train_size//batch_size)
        model.save_weights(path_out +'cnn.h5')
    np.save(path_out + 'history.npy', history.history)


def network_predict(car_name, model):
    paths = path_in + 'test/' + car_name + '_'
    paths = [paths + str(j + 1).zfill(2) + '.jpg' for j in range(16)]
    test = imread_collection(paths).concatenate()
    test = downscale_local_mean(test, (1, scale, scale, 1))
    test -= test.mean()
    test /= test.std()
    masks = model.predict(test, batch_size=8, verbose=1) > 0.5
    return masks[:, :, :, 0]


if __name__ == '__main__':
    network_learning(valid=False)