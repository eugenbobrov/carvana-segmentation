#!/usr/bin/env python3
import os
import numpy as np
import numpy.random as npr
import cv2
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from parameters import path_in

img_id = npr.choice(os.listdir(path_in + 'train/')).split('_')[0]
print(img_id)
#img_id = '0cdf5b5d0ce1'
debug = True
num = 4

fname1 = path_in + 'train/' + img_id+ '_{:0>2}.jpg'.format(num)
fname2 = path_in + 'train/' + img_id+ '_{:0>2}.jpg'.format((num) % 16+1)
img_1_orig = cv2.imread(fname1)
h, w = img_1_orig.shape[0], img_1_orig.shape[1],
img_1_scaled = cv2.resize(img_1_orig, (w//4, h//4))

img_2_orig = cv2.imread(fname2)
h, w = img_2_orig.shape[0], img_2_orig.shape[1],
img_2_scaled = cv2.resize(img_2_orig, (w//4, h//4))

if debug:
    plt.figure()
    plt.subplot(121)
    plt.title('Current image [{}]'.format(num))
    plt.imshow(img_1_scaled)
    plt.subplot(122)
    plt.title('Next image [{}]'.format((num) % 16+1))
    plt.imshow(img_2_scaled)
    plt.savefig('curnext.png')
    plt.show()

img1 = cv2.cvtColor(img_1_scaled, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_2_scaled, cv2.COLOR_BGR2GRAY)
score, dimg = compare_ssim(img1, img2, full=True)
dimg = (dimg * 255).astype("uint8")

thresh = cv2.threshold(dimg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
if debug:
    plt.figure()
    plt.title('Difference image')
    plt.imshow(dimg>thresh)
    plt.savefig('diff.png')
    plt.show()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ROIS = []
for c in cnts[1]:
    (x, y, w, h) = cv2.boundingRect(c)
    if w*h > img1.shape[0]*img1.shape[1]//9:
        ROIS.append([x, y, x+w, y+h])
ROIS = np.array(ROIS)

x1 = ROIS[:,0].min()
y1 = ROIS[:,1].min()

x2 = ROIS[:,2].max()
y2 = ROIS[:,3].max()

if debug:
    plt.figure()
    plt.imshow(img_1_orig)
    rect = Rectangle((x1*4, y1*4), (x2-x1)*4, (y2-y1)*4, fill=False, color='red')
    plt.axes().add_patch(rect)
    plt.savefig('box.png')
    plt.show()

    plt.figure()
    plt.imshow(img_1_orig[y1*4:y2*4, x1*4:x2*4])
    plt.savefig('crop.png')
    plt.show()