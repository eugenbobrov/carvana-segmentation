#!/usr/bin/env python3

path = TRAIN_DIR + id_list[0].split('_')[0] + '_'
path =  [path + str(j + 1).zfill(2) + '.jpg' for j in range(16)]
data = imread_collection(path).concatenate().mean(3)
data -= data.mean(0)



