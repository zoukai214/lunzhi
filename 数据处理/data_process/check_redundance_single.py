
# txt = "/home/hq/workspace/RoP/label/RoP_CrossValidation/cross_val_all_revised.txt"
txt = "/home/hq/workspace/RoP/label/RoP_Plus/plus_train.txt"

img_list = []
redundance_list = []
cnt = 0
# redundance = 0
with open(txt, 'rb') as f:
    for line in f:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        label = int(line.split(',')[1])
        if img_name not in img_list:
            img_list.append(img_name)
        else:
            redundance_list.append(img_name)
            # redundance += 1
        cnt += 1

print('summarizing %s'%txt)
print('num: %d' % cnt)
print('redundance num: %d' % len(redundance_list))
for ele in redundance_list:
    print(ele)