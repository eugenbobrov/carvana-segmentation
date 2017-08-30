#!/usr/bin/env python3
import os
import numpy as np
import multiprocessing as mp
from functools import partial
from skimage.measure import compare_ssim, find_contours
from skimage.transform import downscale_local_mean
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_otsu
from skimage.io import imread
from parameters import path_in, path_out


def make_idx_rois(idx, directory):
    path = path_in + directory + '/' + idx
    for j in range(1, 17):
        name1  = path + '_{:0>2}.jpg'.format(j)
        img1 = imread(name1, as_grey=True)
        img1 = downscale_local_mean(img1, (4, 4))

        name2 = path + '_{:0>2}.jpg'.format((j) % 16+1)
        img2 = imread(name2, as_grey=True)
        img2 = downscale_local_mean(img2, (4, 4))

        grad = compare_ssim(img1, img2, full=True)[1]
        thresh = grad < threshold_otsu(grad)
        thresh = convex_hull_image(thresh)
        cnts = find_contours(thresh, level=0)
        box = cnts[np.argmax(map(len, cnts))].astype('uint16')*4

        x1 = box[:,0].min()
        y1 = box[:,1].min()
        x2 = box[:,0].max()
        y2 = box[:,1].max()

        name = path_out + directory + '_rois/' + idx + '_{:0>2}'.format(j)
        np.save(name, (x1, x2, y1, y2))


def make_rois(directory):
    path = path_out + directory + '_rois/'
    if not os.path.exists(path):
        os.mkdir(path)
    data_path = path_in + directory +'/'
    names = set([s.split('_')[0] for s in os.listdir(data_path)])
    with mp.Pool() as pool:
        pool.map(partial(make_idx_rois, directory=directory), names)


if __name__ == '__main__':
    make_rois('train')
    make_rois('valid')
    make_rois('test')