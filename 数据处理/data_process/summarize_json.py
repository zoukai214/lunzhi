import os
import json


def summarize_json(img_dir, json_dir):
    name_dict = {}
    label_dict = {}
    region_dict = {}
    img_list = os.listdir(img_dir)

    duplicate_list = []
    missing_list = []
    for sets in os.listdir(json_dir):
        data_all = json.load(open(json_dir + sets))

        for data_key in data_all:
            data = data_all[data_key]
            img_label = data['image_labels']
            img_name = data['filename']
            img_region = data['regions']

            if img_name not in name_dict:
                if img_name in img_list:
                    name_dict[img_name] = {'labels': [], 'regions': []}
                else:
                    missing_list.append(img_name)
                    continue
            else:
                duplicate_list.append(img_name)
                continue

            for label_key in img_label:
                label = img_label[label_key]
                if label == '':
                    continue

                name_dict[img_name]['labels'].append(label)
                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append(img_name)

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
                    region_dict[region_label][img_name] = [region_label, shape_name, points]
                    name_dict[img_name]['regions'].append([region_label, shape_name, points])
                except:
                    pass
    return duplicate_list, missing_list, name_dict, label_dict, region_dict


img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
json_dir = '/home/ubuntu/skin_demo/RoP/label/ROP_JSON/'

duplicate_list, missing_list, name_dict, label_dict, region_dict = summarize_json(img_dir, json_dir)

cnt_all = len(name_dict) + len(duplicate_list) + len(missing_list)
print(('Num of labels:{}\n'
       'Num of duplicates:{}\n'
       'Num of not exist file:{}\n'
       'Num of available image:{}\n').format(cnt_all, len(duplicate_list), len(missing_list), len(name_dict)))

for key in label_dict:
    label = label_dict[key]
    print(('Num of label {}:{}').format(key, len(label)))

for key in region_dict:
    region = region_dict[key]
    print(('Num of region {}:{}').format(key, len(region)))
