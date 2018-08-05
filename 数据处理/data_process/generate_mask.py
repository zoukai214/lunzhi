from PIL import Image, ImageDraw
import numpy as np
import csv
import random
import os

imgpath = '/home/ubuntu/skin_demo/Tooth/data/formal/screencut/'
imgpathout2 = '/home/ubuntu/skin_demo/Tooth/data/formal/screencut_mask/'
imgpathout3 = '/home/ubuntu/skin_demo/Tooth/data/formal/screencut_mask_line/'
filename = './teeth_annotation.csv'

def get_data_list(file):
    with open(file) as f:
        reader = csv.reader(f)

        data_list = []

        for row in reader:
            row_list = []
            label_num = int(row[0])
            doc_num = int(row[1])
            patient_num = int(row[2])
            data = row[3]
            date = row[4]

            points_set = []
            label_set = []
            info_list = data.split(',')
            cnt = 0
            skip = False
            for i in range(len(info_list)):
                if skip:
                    skip = False
                    continue

                e = info_list[i].strip('[')
                e = e.strip(']')
                e = e.strip('"')

                if cnt == 0:
                    teeth_label = []
                    teeth_points = []
                    teeth_num = int(e)
                    teeth_label.append(teeth_num)
                if cnt == 1:
                    teeth_position = e
                    teeth_label.append(teeth_position)
                if cnt > 1:
                    if e == 'no' or e == 'yes':
                        broken = e
                        teeth_label.append(broken)
                        points_set.append(tuple(teeth_points))
                        label_set.append(teeth_label)

                        cnt = 0
                        continue
                    else:
                        point = []
                        point.append(int(e))
                        e1 = info_list[i + 1].strip('[')
                        e1 = e1.strip(']')
                        e1 = e1.strip('"')
                        point.append(int(e1))
                        teeth_points.append(tuple(point))
                        skip = True
                cnt = cnt + 1

            row_list.append(label_num)
            row_list.append(doc_num)
            row_list.append(patient_num)
            row_list.append(points_set)
            row_list.append(label_set)
            row_list.append(date)
            data_list.append(row_list)
    return data_list


def get_name_list():
    map_file = './teeth_info.csv'
    name_list = []
    name_list.append('')
    with open(map_file) as f:
        reader = csv.reader(f)
        for row in reader:
            name_list.append(row[1])
    return name_list

def save_mask2():
    name_list = get_name_list()
    data_list = get_data_list(filename)
    img_list = os.listdir(imgpath)

    for row in data_list:
        patient_num = row[2]
        label_num = row[0]
        doc_num = row[1]
        patient_num = row[2]
        points_set = row[3]
        date = row[4]

        img_name = name_list[patient_num]
        if img_name not in img_list or doc_num != 47:
            continue

        im = Image.open(imgpath + img_name)
        width, height = im.size
        im = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(im)
        for teeth in points_set:
            draw.polygon(teeth, fill=(255, 255, 255))
        # for teeth in points_set:
        #     for i in range(len(teeth)):
        #         p0 = teeth[i]
        #         if i + 1 == len(teeth):
        #             p1 = teeth[0]
        #         else:
        #             p1 = teeth[i + 1]

        #         draw.line([p0, p1], fill=(0, 0, 255), width=5)
        # im = im.convert("P")
        output_name = imgpathout2 + img_name.split('.jpg')[0] + '_mask.png'
        print('Mask saved at %s' % output_name)
        im.save(output_name, 'PNG')

def save_mask3():
    name_list = get_name_list()
    data_list = get_data_list(filename)
    img_list = os.listdir(imgpath)

    for row in data_list:
        patient_num = row[2]
        label_num = row[0]
        doc_num = row[1]
        patient_num = row[2]
        points_set = row[3]
        date = row[4]

        img_name = name_list[patient_num]
        if img_name not in img_list or doc_num != 47:
            continue

        im = Image.open(imgpath + img_name)
        width, height = im.size
        im = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(im)
        for teeth in points_set:
            draw.polygon(teeth, fill=(255, 255, 255))
        for teeth in points_set:
            for i in range(len(teeth)):
                p0 = teeth[i]
                if i + 1 == len(teeth):
                    p1 = teeth[0]
                else:
                    p1 = teeth[i + 1]

                draw.line([p0, p1], fill=(0, 0, 255), width=5)
        im = im.convert("P")
        output_name = imgpathout3 + img_name.split('.jpg')[0] + '_mask.png'
        print('Mask saved at %s' % output_name)
        im.save(output_name, 'PNG')


def write_txt_2():
    file_list = get_name_list()
    name_list = []
    data_list = get_data_list(filename)
    img_list = os.listdir(imgpath)
    mask_list = os.listdir(imgpathout2)

    visited_list = []

    for row in data_list:
        patient_num = row[2]
        img_name = file_list[patient_num]
        mask_name = img_name.split('.jpg')[0] + '_mask.png'

        if (img_name in visited_list) or (img_name not in img_list) or (mask_name not in mask_list):
            continue
            
        visited_list.append(img_name)
        name_split = img_name.split('.')
        name = ''
        for i in range(len(name_split)):
            if i<len(name_split)-1:
                name += name_split[i]

        name_list.append(name)


    random.shuffle(name_list)
    train_list = []
    test_list = []
    print(len(name_list))
    thresh = int(len(name_list)*0.8)
    for i in range(len(name_list)):
        if i<thresh:
            train_list.append(name_list[i])
        else:
            test_list.append(name_list[i])

    with open("train2.txt", "w") as f:
        for e in train_list:
            f.write(e+'\n')

    with open("test2.txt", "w") as f:
        for e in test_list:
            f.write(e+'\n')

def write_txt_3():
    file_list = get_name_list()
    name_list = []
    data_list = get_data_list(filename)
    img_list = os.listdir(imgpath)
    mask_list = os.listdir(imgpathout3)

    visited_list = []

    for row in data_list:
        patient_num = row[2]
        img_name = file_list[patient_num]
        mask_name = img_name.split('.jpg')[0] + '_mask.png'

        if (img_name in visited_list) or (img_name not in img_list) or (mask_name not in mask_list):
            continue
            
        visited_list.append(img_name)
        name_split = img_name.split('.')
        name = ''
        for i in range(len(name_split)):
            if i<len(name_split)-1:
                name += name_split[i]

        name_list.append(name)


    random.shuffle(name_list)
    train_list = []
    test_list = []
    print(len(name_list))
    thresh = int(len(name_list)*0.8)
    for i in range(len(name_list)):
        if i<thresh:
            train_list.append(name_list[i])
        else:
            test_list.append(name_list[i])

    with open("train3.txt", "w") as f:
        for e in train_list:
            f.write(e+'\n')

    with open("test3.txt", "w") as f:
        for e in test_list:
            f.write(e+'\n')

# save_mask2()
# write_txt_2()
write_txt_3()
# save_mask()