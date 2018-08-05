import csv
import os

img_path = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
csv_path = '/home/hq/workspace/RoP/master/save/val_rop_2cls_balance_T4/val_result.csv'
output_path = '/home/hq/workspace/RoP/master/save/val_rop_2cls_balance_T4/'

with open(csv_path) as file:
    reader = csv.reader(file)

    data = []
    for row in reader:
        img_name = row[0]
        predict = int(row[1])
        truth = int(row[2])

        if predict != truth:
            img_dir = os.path.join(img_path, img_name)
            new_name = ('{}_predict_{}_truth{}.jpg').format(img_name.split('.')[0], str(predict), str(truth))
            output_dir = os.path.join(output_path, new_name)
            cmd = ('cp {} {}').format(img_dir, output_dir)
            os.system(cmd)

