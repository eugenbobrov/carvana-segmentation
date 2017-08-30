#!/usr/bin/env python3
import os
import numpy as np
import numpy.random as npr
from skimage.io import imread
from skimage.color import rgb2hsv
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from networks import unet
from parameters import path_in, path_out, train_size, valid_size, height, width


def dice_coef(T, P):
    return (2*K.sum(T*P) + 1)/(K.sum(T) + K.sum(P) + 1)


def bce_dice_loss(T, P):
    return binary_crossentropy(T, P)/2 - dice_coef(T, P) + 1


def expand_multiple(x):
    h, w = x.shape[:2]
    hn, wn = (16 - h%16)%16, (16 - w%16)%16
    ht, hd, wl, wr = hn//2 + hn%2, hn//2, wn//2 + wn%2, wn//2

    return np.vstack((

        np.hstack((x[:ht, :wl][::-1, ::-1], x[:ht, :][::-1, :],
                   x[:ht, w - wr:][::-1, ::-1])),
        np.hstack((x[:, :wl][:, ::-1], x, x[:, w - wr:][:, ::-1])),

        np.hstack((x[h - hd:, :wl][::-1, ::-1], x[h - hd:, :][::-1, :],
                   x[h - hd:, w - wr:][::-1, ::-1]))
    ))


def data_generator(directory, data_size):
    rois_path = path_in + directory + '_rois/'
    data_path = path_in + directory +'/'
    mask_path = path_in + directory + '_masks/'
    rois_names = sorted(os.listdir(rois_path))
    data_names = sorted(os.listdir(data_path))
    mask_names = sorted(os.listdir(mask_path))
    while True:
        idx = npr.randint(data_size)
        rois = np.load(rois_path + rois_names[idx])

        data = imread(data_path + data_names[idx])
        data = rgb2hsv(data[rois[0]:rois[1], rois[2]:rois[3]])
        mask = imread(mask_path + mask_names[idx])
        mask = mask[rois[0]:rois[1], rois[2]:rois[3]]/255

        data, mask = expand_multiple(data), expand_multiple(mask)
        yield data[None, :, :, :], mask[None, :, :, None]


def network_learning():
    model = unet()
    model.compile('adam', bce_dice_loss, [dice_coef])
    history = model.fit_generator(epochs=10,
        generator=data_generator('train', train_size),
        steps_per_epoch=train_size,
        validation_data=data_generator('valid', valid_size),
        validation_steps=valid_size,
        callbacks=[EarlyStopping(), ModelCheckpoint(path_out + 'weights.h5',
                   save_best_only=True, save_weights_only=True)])
    np.save(path_out + 'history.npy', history.history)


def network_predict(data_path, rois_path, model):
    rois, data = np.load(rois_path), imread(data_path)
    data = data[rois[0]:rois[1], rois[2]:rois[3], :]
    data = rgb2hsv(data)[None, :, :, :]

    h, w = data.shape[:2]
    hn, wn = (16 - h%16)%16, (16 - w%16)%16
    ht, hd, wl, wr = hn//2 + hn%2, hn//2, wn//2 + wn%2, wn//2

    data = expand_multiple(data)

    mask = np.zeros((height, width), bool)
    pred = model.predict(data, batch_size=1) > 0.5
    mask[rois[0]:rois[1], rois[2]:rois[3]] = pred[0, ht:h - hd, wl:w - wr, 0]
    return mask


if __name__ == '__main__':
    network_learning()