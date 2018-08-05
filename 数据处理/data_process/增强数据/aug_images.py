import os
import numpy as np
from imgaug import augmenters as iaa
from datetime import datetime
import cv2

img_dir = '/home/ubuntu/yangwenbin/chexnet-master/database'
output_dir = '/home/ubuntu/yangwenbin/chexnet-master/database/aug_images'
txt_in = "/home/ubuntu/yangwenbin/chexnet-master/dataset/zzkk.txt"
txt_output = "/home/ubuntu/yangwenbin/chexnet-master/Label/plus_train_balanced.txt"

#images aug
fliplr = aug = iaa.Fliplr(1)

#read original txt,get img and label
with open(txt_in,'r') as fw:
    Inputlines = fw.read().split('\n')

    print('InputLenght',len(Inputlines))

#write label and plot img
with open(txt_output,'a') as fw:
    for line in Inputlines:
        stamp = str(int(round(time.time() * 1000)))
        img_name = line.split(' ')[0]
        img_label = line.split(' ')[1:]
        #a = os.path.join(img_dir,"images_004/00008468_012.png")
        img = cv2.imread(os.path.join(img_dir,img_name))
        img_array = [np.array(img)]
        new_name = img_name.split('/')[1] + '_' + stamp + '.jpg'
        output_path = os.path.join(output_dir,new_name)
        images_aug = fliplr.augment_images(img_array)
        cv2.imwrite(output_path,images_aug[0])





