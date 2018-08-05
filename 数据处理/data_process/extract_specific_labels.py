
origin_txt = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_4cls_train_balanced.txt'
output_txt = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls_train_balanced.txt'

data = []
with open(origin_txt, 'rb') as file:
    for line in file:
        line = line.strip('\n')
        list = line.split(',')
        list[1] = int(list[1])
        if list[1] == 0 or list[1] == 1:
            list[1] = 0
        elif list[1] == 2:
            list[1] = 1
        elif list[1] == 3:
            list[1] = 2

        ele = list[0]+','+str(list[1])
        data.append(ele)

with open(output_txt, "w") as f:
    for e in data:
        f.write(e + '\n')