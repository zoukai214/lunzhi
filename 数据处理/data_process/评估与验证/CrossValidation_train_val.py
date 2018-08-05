import random
import os

txt_path = '/home/hq/workspace/RoP/label/RoP_combine_12_34/all_.txt'
output_dir = '/home/hq/workspace/RoP/label/RoP_combine_12_34/cross_validation/'

data = []
with open(txt_path, 'rb') as file_txt:
    for line in file_txt:
        line = line.strip('\n')
        data.append(line)
random.shuffle(data)

val_percent = 0.2
fold_num = int(1 / val_percent)

dataset_list = []
for i in range(fold_num):
    dataset_list.append([[], []]) #first list is train_list, second is val_list

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
        for line in dataset_list[i][0]:
            file.write(line + '\n')

    with open(val_path, 'a') as file:
        for line in dataset_list[i][1]:
            file.write(line + '\n')
