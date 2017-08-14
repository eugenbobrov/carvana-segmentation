#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 23:23:46 2017

@author: eugen
"""
import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread_collection
import cv2

TRAIN_DIR = 'data_all/train/'
X_SHAPE = (1280, 1918, 3)

id_list = []
for root, subdirs, files in os.walk(TRAIN_DIR):
    for file in files:
        id_list.append(file.split('.')[0])
id_list.sort()
print("Num of images: " + str(len(id_list)))

def make_mask(id):
    id = id.split('_')[0]
    min_img = np.full(X_SHAPE, 255)
    max_img = np.zeros(X_SHAPE)
    for i in range(16):
        if i < 9:
            full_id = id + '_0' + str(i+1)
        else:
            full_id = id + '_' + str(i+1)

        path = TRAIN_DIR + '/' + full_id + '.jpg'
        img = cv2.imread(path)

        min_img = np.minimum(min_img, img)
        max_img = np.maximum(max_img, img)

    return np.mean(max_img - min_img, axis=2)

diff_mag = make_mask(id_list[0*16])
plt.figure(figsize=(15,10))
plt.imshow(diff_mag, cmap='gray', vmin=0, vmax=255)

path = TRAIN_DIR + id_list[0].split('_')[0] + '_'
path =  [path + str(j + 1).zfill(2) + '.jpg' for j in range(16)]
data = imread_collection(path).concatenate().mean(3)
data -= data.mean(0)



