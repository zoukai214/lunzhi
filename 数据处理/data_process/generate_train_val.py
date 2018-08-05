import random

txt_path = '/home/ubuntu/skin_demo/RoP/label/ROP_Phase/json_all_phases_3cls_combine_1_2.txt'
train_txt = '/home/ubuntu/skin_demo/RoP/label/ROP_Phase/json_train_phases_3cls_combine_1_2.txt'
val_txt = '/home/ubuntu/skin_demo/RoP/label/ROP_Phase/json_val_phases_3cls_combine_1_2.txt'

data = []
with open(txt_path, 'rb') as file_txt:
    for line in file_txt:
        line = line.strip('\n')
        data.append(line)
random.shuffle(data)
thresh = int(len(data)*0.9)
train_list = []
test_list = []

for i in range(len(data)):
    if i < thresh:
        train_list.append(data[i])
    else:
        test_list.append(data[i])

with open(train_txt, "w") as f:
    for e in train_list:
        f.write(e + '\n')

with open(val_txt, "w") as f:
    for e in test_list:
        f.write(e + '\n')
