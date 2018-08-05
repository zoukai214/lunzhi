
train_txt = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_2cls_balanced.txt"
val_txt = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_2cls_no_enhance_val.txt"
output = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_2cls_balanced_train.txt'

val_list = []
with open(val_txt, 'rb') as file:
    for line in file:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        val_list.append(img_name)

revised_list = []
with open(train_txt, 'rb') as file:
    for line in file:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        if img_name not in val_list:
            revised_list.append(line)

print('trian num: %d' % len(revised_list))
print('val num: %d' % len(val_list))

with open(output, 'a') as file:
    for line in revised_list:
        file.write(line + '\n')
