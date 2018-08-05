
train_txt = '/home/hq/workspace/RoP/label/RoP_combine_12_34/cross_validation/cross_val_0_train_unbalanced.txt'
val_txt = '/home/hq/workspace/RoP/label/RoP_combine_12_34/cross_validation/cross_val_0_val_unbalanced.txt'

img_list = []
cnt_val = 0
with open(val_txt, 'rb') as f:
    for line in f:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        label = int(line.split(',')[1])
        img_list.append(img_name)
        cnt_val += 1

cnt_train = 0
redundance = 0
with open(train_txt, 'rb') as f:
    for line in f:
        cnt_train += 1
        line = line.strip('\n')
        img_name = line.split(',')[0]
        label = int(line.split(',')[1])
        if img_name in img_list:
            redundance += 1

print(('comparing {} and {}').format(train_txt, val_txt))
print('val num: %d' % cnt_val)
print('train num: %d' % cnt_train)
print('redundance num: %d' % redundance)
