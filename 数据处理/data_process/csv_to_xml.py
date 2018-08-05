#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys


import time
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from xml.etree.ElementTree import Element
from PIL import Image
import os
import json
# from easydict impor EasyDict as edict
from easydict import EasyDict as edict

_classlist = {}


def dict_to_xml(tag, d):
    # '''
    # Turn a simple dict of key/value pairs into XML
    # '''
    elem = Element(tag)
    for key, val in d.items():
        # print(key,val)
        child = Element(key)
        child.text = str(val)
        # print(str(val))
        elem.append(child)

    return elem


def out_xml(root, out_file):
    # """格式化root转换为xml文件"""
    rough_string = ET.tostring(root, "GB2312")
    reared_content = minidom.parseString(rough_string)
    with open(out_file, 'w+') as fs:
        reared_content.writexml(fs, addindent="\t", newl="\n", encoding="GB2312")
    # print('write {} success'.format(out_file))
    return True


import csv

csv_path = '/home/kenneth/zoukai/psoriasis/anno_data/psoriasis_boudingbox.csv'
# csv_path = '/home/kenneth/zoukai/psoriasis/anno_data/biaozhu_psoriasismask_anno.csv'
# csv_path = '/home/kenneth/hq/workspace/yolo2-pytorch-master/DRMASK20180601.csv'
# img_dir = '/home/kenneth/hq/workspace/Faster_Rcnn_detect/data/imgs'
img_dir = '/home/kenneth/zoukai/psoriasis/img_data/JPGEImages/'
# img_output_dir = '/home/kenneth/hq/workspace/Faster_Rcnn_detect/data/imgs'
# img_dir = '/home/kenneth/QR/ChexNet/chexnet-master/database/images_dicom'
# ome/kenneth/zoukai/psoriasis/img_data/Annotations
output_path = '/home/kenneth/zoukai/psoriasis/img_data/Annotations/'

data_dict = {}
repeat_cnt = 0
with open(csv_path) as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        # img_name = row[0]
        # label = row[1]
        # xmin = float(row[2])
        # ymin = float(row[3])
        # w = float(row[4])
        # h = float(row[5])
        # xmax = xmin + w
        # ymax = ymin + h
        #
        # img = Image.open(os.path.join(img_dir, img_name))
        # img.convert('RGB')
        # w, h = img.size
        # img_name = '{}.jpg'.format(img_name.split('.')[0])
        # try:
        #     img.save(os.path.join(img_output_dir, img_name))
        # except:
        #     print('Failure!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{}'.format(img_name))
        img_name = row[0]
        # print(img_name)
        annotation_raw = row[1]
        # print(annotation_raw)
        annotation = json.loads(annotation_raw, encoding="GB2312")
        # position = annotation[0]
        # print(position)
        print('lengh of position:{}'.format(len(annotation)))
        if len(annotation) == 0:
            continue

        inpath = '/home/kenneth/zoukai/psoriasis/img_data/JPGEImages/{}'.format(img_name)
        unipath = unicode(inpath, "GB2312")
        img = Image.open(unipath)

        w = img.size[0]
        # print(w)
        h = img.size[1]

        base = {'folder': 'VOC2007', 'filename': img_name, 'segmented': 0}
        size = {'width': w, 'height': h, 'depth': 3}
        size_node = dict_to_xml('size', size)
        root = dict_to_xml('annotation', base)
        # print(root[0])
        if img_name not in data_dict:
            root.append(size_node)
            data_dict[img_name] = root
            # print(data_dict[img_name])

        for i in range(len(annotation)):

            label = annotation[i][0]
            print(label)
            if label not in _classlist:
                _classlist[label] = 1
            else:
                _classlist[label] += 1
            print(annotation[i])
            data = annotation[i][1]
            point = data[0]
            delta = data[1]
            ori_x = point[0]
            ori_y = point[1]
            delta_x = delta[0]
            delta_y = delta[1]
            if delta_x < 0:
                xmax = ori_x
                xmin = ori_x + delta_x
            else:
                xmin = ori_x
                xmax = ori_x + delta_x
            if delta_y < 0:
                ymax = ori_y
                ymin = ori_y + delta_y
            else:
                ymin = ori_y
                ymax = ori_y + delta_y

            object = {'name': label,
                      'pose': 'Frontal',
                      'truncated': 0,
                      'difficult': 0}
            object_node = dict_to_xml('object', object)

            bndbox = {'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax)}
            bndbox_node = dict_to_xml('bndbox', bndbox)
            object_node.append(bndbox_node)
            data_dict[img_name].append(object_node)
            # print(img_name,data_dict[img_name])
            # break
            # print('read row {} success'.format(i))

for img_name in data_dict:
    root = data_dict[img_name]
    (file_name, extension) = os.path.splitext(img_name)
    # print(file_name)
    # xml_name = '{}.xml'.format(img_name.split('.')[0])
    xml_name = '{}.xml'.format(file_name)
    out_xml(root, os.path.join(output_path, xml_name))
# print(_classlist)