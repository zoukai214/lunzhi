import random
import os

txt_path = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls.txt'
output_dir = '/home/hq/workspace/RoP/label/RoP_3cls_CrossValidation/'

label_dict = {}
# data = []
with open(txt_path, 'rb') as file_txt:
    for line in file_txt:
        line = line.strip('\n')
        split = line.split(',')
        img_name = split[0]
        label = int(split[1])
        if label not in label_dict:
            label_dict[label] = []

        label_dict[label].append(line)


val_percent = 0.2
fold_num = int(1 / val_percent)

dataset_list = []
for i in range(fold_num):
    dataset_list.append([[], []]) #first list is train_list, second is val_list

for key in label_dict:
    data = label_dict[key]
    for i in range(len(data)):
        for j in range(fold_num):
            if i > j * val_percent * len(data) and i <= (j + 1) * val_percent * len(data):
                dataset_list[j][1].append(data[i])
            else:
                dataset_list[j][0].append(data[i])

for i in range(fold_num):
    train_name = ('cross_val_{}_train_unbalanced.txt').format(i)
    val_name = ('cross_val_{}_val_unbalanced.txt').format(i)
    train_path = os.path.join(output_dir, train_name)
    val_path = os.path.join(output_dir, val_name)

    with open(train_path, 'a') as file:
        train_data = dataset_list[i][0]
        random.shuffle(train_data)
        for line in train_data:
            file.write(line + '\n')

    with open(val_path, 'a') as file:
        val_data = dataset_list[i][1]
        random.shuffle(val_data)
        for line in val_data:
            file.write(line + '\n')
