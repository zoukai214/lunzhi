import os
import json
from pprint import pprint
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import pandas
import numpy as np

img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
json_dir = '/home/ubuntu/skin_demo/RoP/label/ROP_JSON/'
label_dict = {}
region_dict = {}

img_list = os.listdir(img_dir)

cnt_all, cnt_duplicate, cnt_not_exist = 0, 0, 0
img_name_list = []
duplicate_list = []

cls_0_list, cls_1_list = [], []
for sets in os.listdir(json_dir):
    data_all = json.load(open(json_dir + sets))
    cnt_all += len(data_all)

    for data_key in data_all:
        data = data_all[data_key]
        img_label = data['image_labels']
        img_name = data['filename']
        img_region = data['regions']

        if img_name not in img_name_list:
            if img_name in img_list:
                img_name_list.append(img_name)
            else:
                cnt_not_exist += 1
                continue
        else:
            cnt_duplicate += 1
            continue

        if_cls_1 = False
        for label_key in img_label:
            label = img_label[label_key]
            if label == '':
                continue
            if label not in label_dict:
                label_dict[label] = []
            if img_name not in label_dict[label]:
                label_dict[label].append(img_name)

            if label == 'D002.A001.P001' or label == 'D002.A001.P002' or label == 'D002.A001.P003':
                if_cls_1 = True

        if if_cls_1:
            cls_1_list.append(img_name)
        else:
            cls_0_list.append(img_name)

        for region_key in img_region:
            region = img_region[region_key]
            shape_attributes = region['shape_attributes']
            region_attributes = region['region_attributes']

            try:
                region_label = region_attributes.popitem()[1]
                if region_label not in region_dict:
                    region_dict[region_label] = {}

                shape_name = shape_attributes['name']
                points = []
                try:
                    cx = shape_attributes['cx']
                    cy = shape_attributes['cy']
                    center = (cx, cy)
                    points.append(center)
                except:
                    all_x = shape_attributes['all_points_x']
                    all_y = shape_attributes['all_points_y']
                    for x, y in zip(all_x, all_y):
                        points.append((x, y))

                    region_dict[region_label][img_name] = [shape_name, points]

            except:
                pass

# ========================================================================================================
print(('Num of labels:{}\nNum of duplicates:{}\nNum of not exist file:{}\n'
       'Num of available image:{}\n').format(cnt_all, cnt_duplicate, cnt_not_exist, len(img_name_list)))

for key in label_dict:
    label = label_dict[key]
    print(('Num of label {}:{}').format(key, len(label)))

for key in region_dict:
    region = region_dict[key]
    print(('Num of region {}:{}').format(key, len(region)))
# =========================================output to specific label to txt=========================================

txt_path = '/home/hq/workspace/RoP/label/RoP_Plus/plus_all.txt'

with open(txt_path, 'w') as file:
    for img_name in cls_0_list:
        file.write(('{},{}\n').format(img_name, 0))
    for img_name in cls_1_list:
        file.write(('{},{}\n').format(img_name, 1))
print(('Num of label 0:{}\nNum of label 1:{}').format(len(cls_0_list), len(cls_1_list)))
print('labels stored at %s' % txt_path)
