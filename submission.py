#!/usr/bin/env python3
import os
import time
import numpy as np
from inference import network_predict
from keras.utils import load_model
from parameters import path_in, path_out, test_size


def rle(img):
    starts = np.array((img[:-1] == 0) & (img[1:] == 1))
    ends = np.array((img[:-1] == 1) & (img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    l = np.vstack((ends_ix, lengths)).ravel('F')
    return ' '.join(str(l)[1:-1].split())


def make_submission():
    data_path = path_in + 'test/'
    rois_path = path_in + 'test_rois/'
    data_names = sorted(os.listdir(data_path))
    rois_names = sorted(os.listdir(rois_path))

    model = load_model('model.h5')
    submission = list(['img,rle_mask'])
    for j, data_name, rois_name in enumerate(zip(data_names, rois_names)):
        clock = time.clock()

        mask = network_predict(data_path + data_name,
            rois_path + rois_name, model).ravel().astype('uint8')
        submission.append(data_name + ',' + rle(mask))

        print(time.clock() - clock, 'sec per loop')
        print('done: {}/{} images'.format((j + 1), test_size))

    np.savetxt(path_out + 'submission.csv', submission, fmt='%s')


if __name__ == '__main__':
    make_submission()