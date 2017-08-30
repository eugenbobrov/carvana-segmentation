#!/usr/bin/env python3
import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from parameters import path_in, path_out
from networks import unet
from inference import network_predict


def image_open(mask, car_id, idx, dir_name):
    path = path_in + dir_name + '/{}_{}.jpg'.format(car_id, idx)
    im_car = imread(path)
    if mask == True:
        path = path_in + dir_name + '_masks/{}_{}_mask.gif'.format(car_id, idx)
        im_mask = imread(path) > 0
        im_car = im_car*im_mask[:, :, None]
    return im_car


def create_random_test_masks():
    name = npr.choice(os.listdir(path_in + 'test')).split('_')[0]

    path = path_out + 'test_masks'
    if not os.path.exists(path):
        os.mkdir(path)
    path += '/' + name + '_'

    model = unet()
    model.load_weights('weights.h5')

    for j in range(1, 17):
        data_path = path_in + 'test/' + name + '_{:0>2}.jpg'.format(j)
        rois_path = path_in + 'test_rois/' + name + '_{:0>2}.npy'.format(j)
        mask = network_predict(data_path, rois_path, model)
        imsave(path + str(j).zfill(2) + '_mask.gif', mask*255)


def draw_random_car_rotation(dir_name='test'):
    car_names = os.listdir(path_in + dir_name + '_masks')
    car_id = npr.choice(car_names).split('_')[0]

    car_img = np.vstack([np.hstack([
        image_open(f, car_id, str(x).zfill(2), dir_name)
        for x in range(j, j + 4)])
        for f, j in zip((False, True)*4, np.repeat((1, 5, 9, 13), 2))])

    car_img = resize(car_img, np.array(car_img.shape[:2])/4, mode='constant')
    imsave(path_out + dir_name + '.jpg', car_img)


def draw_random_car(dir_name='test'):
    car_names = os.listdir(path_in + dir_name + '_masks')
    car_id = npr.choice(car_names).split('_')[0]

    car_path = os.path.join(path_in, dir_name, car_id)
    car = imread(car_path +  '_04.jpg')
    mask_path = os.path.join(path_in, dir_name + '_masks', car_id)
    mask = imread(mask_path + '_04_mask.gif') > 0

    imsave(path_out + 'car.png', car)
    imsave(path_out + 'masked_car.png', car*mask[:, :, None])


def plot_model_history():
    history = np.load(path_in + 'history.npy').tolist()
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(history['loss']); plt.plot(history['val_loss'])
    plt.title('model loss'); plt.legend(['train', 'valid'])
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.savefig(path_out + 'loss.png')

    plt.figure()
    plt.plot(history['dice_coef']); plt.plot(history['val_dice_coef'])
    plt.title('model dice'); plt.legend(['train', 'valid'])
    plt.ylabel('dice'); plt.xlabel('epoch')
    plt.savefig(path_out + 'dice.png')


if __name__ == '__main__':
    create_random_test_masks()