import random
txt_path = '/home/hq/workspace/RoP/label/RoP_Plus/plus_all.txt'
train_txt = '/home/hq/workspace/RoP/label/RoP_Plus/plus_train.txt'
val_txt = '/home/hq/workspace/RoP/label/RoP_Plus/plus_val.txt'

data = []
with open(txt_path, 'rb') as file_txt:
    for line in file_txt:
        line = line.strip('\n')
        data.append(line)
random.shuffle(data)

thresh = int(len(data)*0.9)
train_list = []
val_list = []

for i in range(len(data)):
    if i<thresh:
        train_list.append(data[i])
    else:
        val_list.append(data[i])

with open(train_txt, "w") as f:
    for line in train_list:
        f.write(line + '\n')

with open(val_txt, "w") as f:
    for line in val_list:
        f.write(line + '\n')