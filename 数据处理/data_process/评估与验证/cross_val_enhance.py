# -*- coding: utf-8 -*-
from __future__ import print_function, division
from scipy.stats import itemfreq
import os
import cv2
import numpy as np
import random
import imgaug as ia
from imgaug import augmenters as iaa
import time
from dateutil import tz
from datetime import datetime

img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
output_dir = '/home/hq/workspace/RoP/data/combine_12_34_cross_val'

txt_dir = '/home/hq/workspace/RoP/label/RoP_combine_12_34/cross_validation'
txt_whole_path = '/home/hq/workspace/RoP/label/RoP_combine_12_34/all.txt'
train_txt_list = ['cross_val_0_train_unbalanced.txt',
                  'cross_val_1_train_unbalanced.txt',
                  'cross_val_2_train_unbalanced.txt',
                  'cross_val_3_train_unbalanced.txt',
                  'cross_val_4_train_unbalanced.txt']
val_txt_list = ['cross_val_0_val_unbalanced.txt',
                'cross_val_1_val_unbalanced.txt',
                'cross_val_2_val_unbalanced.txt',
                'cross_val_3_val_unbalanced.txt',
                'cross_val_4_val_unbalanced.txt']

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
    # iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
]

aug_cls0 = [
    iaa.Fliplr(1),
    # iaa.Flipud(1),
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
    # iaa.Sharpen(alpha=(0.6, 0.8), lightness=(0.75, 1.5)),
    # iaa.Sharpen(alpha=(0.8, 1), lightness=(0.75, 1.5)),
]

# =======================prepare for update every train txt for a single enhancement===============================

val_name_list = []
train_data_list = []
for i in range(len(val_txt_list)):
    val_name = []
    with open(os.path.join(txt_dir, val_txt_list[i]), 'rb') as file:
        for line in file:
            line = line.strip('\n')
            name = line.split(',')[0]
            val_name.append(name)
    val_name_list.append(val_name)

    train_data = []
    with open(os.path.join(txt_dir, train_txt_list[i]), 'rb') as file:
        for line in file:
            line = line.strip('\n')
            train_data.append(line)
    train_data_list.append(train_data)

# ============================================================================================

data = []
with open(txt_whole_path, 'rb') as file:
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
                if skip < 0.75:
                    continue

                stamp = str(int(round(time.time() * 1000)))
                time.sleep(0.0005)
                # stamp = str(t).split('.')[1]
                new_name = img_name.split('.')[0] + '_' + stamp + '.jpg'
                output_path = os.path.join(output_dir, new_name)
                images_aug = aug.augment_images(img_array)
                cv2.imwrite(output_path, images_aug[0])
                aug_cnt += 1
                print('augmentation done')

                # update every train txt for a single enhancement
                for j, val_name in enumerate(val_name_list):
                    if img_name not in val_name:
                        train_data = train_data_list[j]

                        new_line = ('{},{}').format(new_name, label)
                        train_data.append(new_line)

        elif label == 1:
            for i, aug in enumerate(aug_cls1):
                skip = random.random()
                if skip < 0.75:
                    continue

                stamp = str(int(round(time.time() * 1000)))
                time.sleep(0.0005)
                new_name = img_name.split('.')[0] + '_' + stamp + '.jpg'
                output_path = os.path.join(output_dir, new_name)
                images_aug = aug.augment_images(img_array)
                cv2.imwrite(output_path, images_aug[0])
                aug_cnt += 1
                print('augmentation done')

                # update every train txt for a single enhancement
                for j, val_name in enumerate(val_name_list):
                    if img_name not in val_name:
                        train_data = train_data_list[j]
                        new_line = ('{},{}').format(new_name, label)
                        train_data.append(new_line)
    except:
        print("augmentation failed: %s" % img_name)
        failed_list.append(img_name)

for i, train_data in enumerate(train_data_list):
    train_txt = train_txt_list[i]
    new_train_txt = ('{}_balanced.txt').format(train_txt.split('.')[0].split('_unbalanced')[0])
    new_txt_path = os.path.join(txt_dir, new_train_txt)

    with open(new_txt_path, 'a') as file:
        for line in train_data:
            file.write(line+'\n')

print(('total augmentation num:{}').format(aug_cnt))