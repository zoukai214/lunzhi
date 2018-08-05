import os

txt_path = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls_combine12.txt'
output_path = '/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls.txt'
img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'

img_list = os.listdir(img_dir)
not_exist = 0
data = []
with open(txt_path, 'rb') as file_txt:
    for line in file_txt:
        line = line.strip('\n')
        img_name = line.split(',')[0]
        label = int(line.split(',')[1])

        if img_name not in img_list:
            not_exist += 1
            continue
        data.append(line)

with open(output_path, 'w') as file:
    for line in data:
        file.write(line + '\n')

print('removed %d not exist items' % not_exist)


