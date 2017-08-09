import os
import time
import numpy as np
from skimage.transform import resize
from skimage.morphology import binary_closing
from inference import get_unet, predict_masks
from parameters import (raw_height, raw_width, path_in,
                       path_out, batch_size, test_size)


def rle(img):
    starts = np.array((img[:-1] == 0) & (img[1:] == 1))
    ends = np.array((img[:-1] == 1) & (img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    l = np.vstack((ends_ix, lengths)).ravel('F')
    return ' '.join(str(l)[1:-1].split())


def make_submission():
    names = set([s.split('_')[0] for s in os.listdir(path_in + 'test')])
    model = get_unet()
    model.load_weights(path_in + 'unet.h5')

    submission = list(['img,rle_mask'])
    for j, name in enumerate(names):
        clock = time.clock()
        masks = predict_masks(name, model)

        for k, mask in enumerate(masks):
            mask = binary_closing(mask)
            mask = resize(mask, (raw_height, raw_width), mode='constant')
            mask = mask.ravel().astype('uint8')
            row = name + '_' + str(k + 1).zfill(2) + '.jpg,' + rle(mask)
            submission.append(row)

        print(time.clock() - clock, 'sec per loop')
        print('done: {}/{} images'.format((j + 1)*batch_size, test_size))

    np.savetxt(path_out + 'submission.csv', submission, fmt='%s')


if __name__ == '__main__':
    make_submission()