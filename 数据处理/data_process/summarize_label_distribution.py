import os

txt_dir = '/home/hq/workspace/RoP/label/RoP_combine_12_34/cross_validation/'
txt_list = ['cross_val_0_train_balanced.txt',
            'cross_val_1_train_balanced.txt',
            'cross_val_2_train_balanced.txt',
            'cross_val_3_train_balanced.txt',
            'cross_val_4_train_balanced.txt',]
img_dir = '/home/hq/workspace/RoP/data/combine_12_34_cross_val'

img_list = os.listdir(img_dir)

for txt_name in txt_list:
    txt_path = os.path.join(txt_dir, txt_name)
    label_dict = {}
    not_exist = 0
    with open(txt_path, 'r') as file_txt:
        for line in file_txt:
            line = line.strip('\n')
            img_name = line.split(',')[0]
            label = int(line.split(',')[1])

            if img_name not in img_list:
                not_exist += 1
                continue

            if label not in label_dict:
                label_dict[label] = 0
            label_dict[label] += 1

    print(txt_path)
    for key in label_dict:
        print(('num of {}: {}').format(key, label_dict[key]))
    print('num of not exist label: %d' % not_exist)
