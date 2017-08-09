import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import binary_closing
from parameters import path_in, path_out, raw_height, raw_width
from inference import get_unet, predict_masks


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

    model = get_unet()
    model.load_weights(path_in + 'unet.h5')
    masks = predict_masks(name, model)

    path = path_out + 'test_masks/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += name + '_'
    for j, mask in enumerate(masks):
        mask = binary_closing(mask)
        mask = resize(mask, (raw_height, raw_width), mode='constant')
        imsave(path + str(j + 1).zfill(2) + '_mask.gif', mask)


def draw_random_car_rotation(dir_name='test'):
    car_names = os.listdir(path_in + dir_name + '_masks')
    car_id = npr.choice(car_names).split('_')[0]

    car_img = np.vstack([np.hstack([
        image_open(f, car_id, str(x).zfill(2), dir_name)
        for x in range(j, j + 4)])
        for f, j in zip((False, True)*4, np.repeat((1, 5, 9, 13), 2))])

    car_img = resize(car_img, (raw_width, raw_height), mode='constant')
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
    loss = np.load(path_in + 'loss.npy')
    dice = np.load(path_in + 'dice.npy')
    accuracy = np.load(path_in + 'accuracy.npy')
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(accuracy)
    plt.title('model train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(path_out + 'accuracy.png')
    plt.figure()
    plt.plot(loss)
    plt.title('model train loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(path_out + 'loss.png')
    plt.figure()
    plt.plot(dice)
    plt.title('model train dice')
    plt.ylabel('dice')
    plt.xlabel('epoch')
    plt.savefig(path_out + 'dice.png')


if __name__ == '__main__':
    create_random_test_masks()
    draw_random_car()