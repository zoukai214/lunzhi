# -*- coding: utf-8 -*-
from __future__ import print_function, division
from scipy.stats import itemfreq
import cv2
import numpy as np
import os
import random
import time
import imgaug as ia
from imgaug import augmenters as iaa
from dateutil import tz
from datetime import datetime

img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced/'
output_dir = '/home/hq/workspace/RoP/data/RoP_Plus'
txt_in = "/home/hq/workspace/RoP/label/RoP_Plus/plus_train.txt"
txt_output = "/home/hq/workspace/RoP/label/RoP_Plus/plus_train_balanced.txt"

aug_cls0 = [
    iaa.Fliplr(1),
    # iaa.Flipud(1),
    # iaa.ContrastNormalization((0.8, 0.9)),
    # iaa.ContrastNormalization((0.9, 1)),
    # iaa.ContrastNormalization((1, 1.2)),
    iaa.ContrastNormalization((1.2, 1.5)),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Add((-40, 40), per_channel=0.7),
    # iaa.Sharpen(alpha=(0.4, 0.6), lightness=(0.75, 1.5)),
    iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    # iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
    # iaa.ContrastNormalization((0.8, 0.9)),
    # iaa.ContrastNormalization((0.9, 1)),
    # iaa.ContrastNormalization((1, 1.2)),
    # iaa.ContrastNormalization((1.2, 1.5)),
    # iaa.Sharpen(alpha=(0.4, 0.6), lightness=(0.75, 1.5)),
    # iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    # iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
]

aug_cls1 = [
    iaa.Fliplr(1),
    iaa.Flipud(1),
    # iaa.ContrastNormalization((0.8, 0.9)),
    # iaa.ContrastNormalization((0.9, 1)),
    iaa.ContrastNormalization((1, 1.2)),
    iaa.ContrastNormalization((1.2, 1.5)),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Add((-40, 40), per_channel=0.7),
    # iaa.Sharpen(alpha=(0.4, 0.6), lightness=(0.75, 1.5)),
    iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    # iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
    # iaa.ContrastNormalization((0.8, 0.9)),
    # iaa.ContrastNormalization((0.9, 1)),
    # iaa.ContrastNormalization((1, 1.2)),
    # iaa.ContrastNormalization((1.2, 1.5)),
    # iaa.Sharpen(alpha=(0.4, 0.6), lightness=(0.75, 1.5)),
    iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
]

data = []
with open(txt_in, 'r') as file:
    for line in file:
        line = line.strip('\n')
        data.append(line)

aug_cnt = 0
failed_list = []

for line in data:
    img_name = line.split(',')[0]
    label = int(line.split(',')[1])

    img = cv2.imread(os.path.join(img_dir, img_name))
    img_array = [np.array(img)]
    cv2.imwrite(os.path.join(output_dir, img_name), img)

    try:
        if label == 0:
            for i, aug in enumerate(aug_cls0):
                skip = random.random()
                if skip < 0.9:
                    continue

                stamp = str(int(round(time.time() * 1000)))
                time.sleep(0.0005)
                # stamp = str(t).split('.')[1]
                new_name = img_name.split('.')[0] + '_' + stamp + '.jpg'
                output_path = os.path.join(output_dir, new_name)
                images_aug = aug.augment_images(img_array)
                cv2.imwrite(output_path, images_aug[0])
                aug_cnt += 1

                new_line = '{},{}'.format(new_name, 0)
                data.append(new_line)
                print('augmentation done')


        elif label == 1:
            for i, aug in enumerate(aug_cls1):
                skip = random.random()
                if skip > 0.1667:
                    continue

                stamp = str(int(round(time.time() * 1000)))
                time.sleep(0.0005)
                new_name = img_name.split('.')[0] + '_' + stamp + '.jpg'
                output_path = os.path.join(output_dir, new_name)
                images_aug = aug.augment_images(img_array)
                cv2.imwrite(output_path, images_aug[0])
                aug_cnt += 1

                new_line = '{},{}'.format(new_name, 1)
                data.append(new_line)
                print('augmentation done')

    except:
        print("augmentation failed: %s" % img_name)
        failed_list.append(img_name)

random.shuffle(data)
with open(txt_output, 'w') as file:
    for line in data:
        file.write(line + '\n')

print(('total augmentation num:{}').format(aug_cnt))
print('Num of failures:{}'.format(len(failed_list)))
for img_name in failed_list:
    print('failed img name:{}'.format(img_name))
