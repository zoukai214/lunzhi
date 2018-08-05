import os

img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
train_dataTXT = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_2cls_train_balanced_revised.txt"
val_dataTXT = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls_only12_val.txt"

train_output = '/home/hq/workspace/RoP/data/train/'
val_output = '/home/hq/workspace/RoP/data/val/'

trian_img_list = []
val_img_list = []

with open(train_dataTXT, 'rb') as file:
    for line in file:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        trian_img_list.append(img_name)
with open(val_dataTXT,'rb') as file:
    for line in file:
        line = line.strip('/n')
        img_name = line.split(',')[0]
        val_img_list.append(img_name)

for img in trian_img_list:
    img_path = os.path.join(img_dir, img)
    os.system('cp ' + img_path + ' ' + train_output)
for img in val_img_list:
    img_path = os.path.join(img_dir, img)
    os.system('cp ' + img_path + ' ' + val_output)
