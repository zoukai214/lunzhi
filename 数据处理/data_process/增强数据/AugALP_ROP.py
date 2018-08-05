# -*- coding: utf-8 -*-
from __future__ import print_function, division
from scipy.stats import itemfreq
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from dateutil import tz
from datetime import datetime


img_dir= '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced/'
output_dir= '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
txt_in='/home/ubuntu/skin_demo/RoP/label/ROP_Phase/jsonAll_phases_4cls_train.txt'
txt_output = '/home/ubuntu/skin_demo/RoP/label/ROP_Phase/jsonAll_phases_4cls_train_balanced.txt'

with open(txt_in, 'r') as fw:
    Inputlines = fw.read().split('\n')
    print('InputLength',len(Inputlines))

a,b=0,1
if b > a:
    aug300 = iaa.Sequential([
        iaa.Sometimes(0.9,
            iaa.OneOf([
                iaa.Affine(rotate=(1, 20)),
                iaa.Affine(rotate=(-20, 1)),
            ]),
        ),

        iaa.Sometimes(0.5,
            iaa.Scale((0.9, 1)),
        ),

        iaa.Sometimes(0.5,
            iaa.OneOf([
                iaa.Add((-40, 40), per_channel=0.5),
            ]),
        ),
    ])

    fliplr = aug = iaa.Fliplr(1)

with open(txt_output, 'a') as fw:
    for line in Inputlines:
        img = line.split(',')[0]
        imglabel = line.split(',')[1]

        readimage = cv2.imread(img_dir+img)
        imageArray = [np.array(readimage)]
        print('done')
        if imglabel == '1' or imglabel == '2':
            images_aug = fliplr.augment_images(imageArray)
            img_name = 'horizon' + '_' + img
            cv2.imwrite(output_dir + img_name, images_aug[0])
            fw.write(img_name +','+ imglabel +'\n')
        elif imglabel == '0':
            images_aug = fliplr.augment_images(imageArray)
            img_name = 'horizon' + '_' + img
            cv2.imwrite(output_dir + img_name, images_aug[0])
            fw.write(img_name +','+ imglabel +'\n')
            a,pershare=0,5
            while a < pershare:
                images_aug = aug300.augment_images(imageArray)
                img_name = 'Aug' + '_' + str(a) + '_'+ img
                cv2.imwrite(output_dir + img_name, images_aug[0])
                fw.write(img_name +','+ imglabel +'\n')
                a += 1
        elif imglabel == '3':
            images_aug = fliplr.augment_images(imageArray)
            img_name = 'horizon' + '_' + img
            cv2.imwrite(output_dir + img_name, images_aug[0])
            fw.write(img_name +','+ imglabel +'\n')
            a,pershare=0,5
            while a < pershare:
                images_aug = aug300.augment_images(imageArray)
                img_name = 'Aug' + '_' + str(a) + '_'+ img
                cv2.imwrite(output_dir + img_name, images_aug[0])
                fw.write(img_name +','+ imglabel +'\n')
                a += 1
        # print(aimg_nameO,age,'period1','pershare',round(pershare),'count',a)
