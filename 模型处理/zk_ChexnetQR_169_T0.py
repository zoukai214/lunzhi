# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
from torch.autograd import Variable
# import torchvision.transforms as transforms
import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import random
import numpy
import time
# import datetime
from dateutil import tz
from datetime import datetime

# from tqdm import tqdm
from scipy.stats import itemfreq

from PIL import ImageFile
# import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics.ranking import roc_auc_score
import numpy as np
import sys
import math
# sys.path.append('./pretrained-models')
# import pretrainedmodels
import DensenetModels

###With Crop data------------------------------


'''
================================= Check 1 train/val  ===================================
'''

'''
--------------- without0 ---------------
'''

img_dir = '/home/kenneth/zoukai/zk_chexnet/database/'
#train_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/without0/train_all.txt"
test_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/without0/val_all.txt"
model_save_dir = "/home/kenneth/QR/ChexNet/chexnet-master/output_models/ChexnetQR_169_T0/"
time1 = datetime.now()

# '''
# --------------- with0 0 ---------------
# '''
#
# img_dir = '/home/kenneth/zoukai/zk_chexnet/database/'
# train_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/with0/train11.txt"
# test_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/with0/test11.txt"
# model_save_dir = "/home/kenneth/QR/ChexNet/chexnet-master/output_models/ChexnetQR_169_T1/"
# time1 = datetime.now()
#
# '''
# --------------- part0 10% ---------------
# '''
#
# img_dir = '/home/kenneth/zoukai/zk_chexnet/database/'
# train_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/part0/train_concat.txt"
# test_dataS = "/home/kenneth/QR/ChexNet/chexnet-master/Label/part0/test_concat.txt"
# model_save_dir = "/home/kenneth/QR/ChexNet/chexnet-master/output_models/ChexnetQR_169_T2/"
# time1 = datetime.now()

'''
================================= Check 2 hyperparameter ===================================
'''
torch.set_num_threads(4)

TryTimes = 'ChexnetQR_169_T0'
modelname = 'Dense_169'
# batchSize = 512*4*5
batchSize = 64
learningRate = 0.001
L2weightDecay = 0.007
# L2weightDecay = 0.02


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
GPU_id = [0, 1]
nnClassCount = 6

#orginal 2592*3456
sizex,sizey=224,224

transform_test = transforms.Compose([
    transforms.Resize([sizex,sizey]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class MyDataset(Dataset):
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        fileDescriptor = open(pathDatasetFile, "r")
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel= torch.LongTensor(self.listImageLabels[index])
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)


#train_data = MyDataset(pathImageDirectory=img_dir,pathDatasetFile = train_dataS, transform = transform_train)
val_data = MyDataset(pathImageDirectory=img_dir,pathDatasetFile = test_dataS, transform = transform_test)
#train_loader = DataLoader(dataset = train_data, batch_size = batchSize, shuffle=True,num_workers=2*len(GPU_id))
test_loader = DataLoader(dataset= val_data, batch_size = batchSize, shuffle=False ,num_workers=2*len(GPU_id))


print('This model using folloing sets:' + '\n'
      'start time: {}'.format(datetime.strftime(datetime.utcnow().replace(tzinfo=tz.gettz('UTC')).\
                                                astimezone(tz.gettz('Asia/Shanghai')), "%Y-%m-%d %H:%M:%S")) + '\n'
      'Try Times: {}'.format(TryTimes) + '\n'
      'model: {}'.format(modelname) + '\n'
      'input size : [{},{}]'.format(sizex,sizey) + '\n'
      'batch size = {}'.format(batchSize) + '\n'
      'L2 weightDecay = {}'.format(L2weightDecay) + '\n'
      'img_dir: {}'.format(img_dir) + '\n'
      'val_data: {}'.format(test_dataS) + '\n'
      'GPU_id: {}'.format(GPU_id) + '\n'
      )

'''
================================= Check 3 Models =================================
'''

def load_pretrained_diff_parameter(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    diff = {k: v for k, v in model_dict.items() if \
            k in pretrained_dict and pretrained_dict[k].size() != v.size()}

    pretrained_dict.update(diff)
    model.load_state_dict(pretrained_dict)
    return model

# ================================= DenseNet201 ======================================
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121(x)
        return x
pathModel = '/home/kenneth/QR/ChexNet/chexnet-master/output_models/ChexnetQR_169_T0/ ChexnetQR_169_T0_Dense_169_epoch38_loss0.490133.pkl'

model = DensenetModels.DenseNet121(nnClassCount, True).cuda()
model = torch.nn.DataParallel(model).cuda()
pretrained_dict = torch.load(pathModel)
model.load_state_dict(pretrained_dict)


AUROC_avg_best = 0
def compute_AUCs(gt, pred,nnClassCount):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    np.savetxt('/home/kenneth/zoukai/f1_output/gt.csv', datanpGT, delimiter=',', fmt="%5.2f")
    np.savetxt('/home/kenneth/zoukai/f1_output/pred.csv', datanpPRED, delimiter=',', fmt='%5.4f')
    for i in range(nnClassCount):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

CLASS_NAMES = ['Atelectasis','Effusion', 'Infiltration','Mass','Nodule', 'Pneumothorax']

model.eval()
enum_batch = 0

gt = torch.FloatTensor()
gt = gt.cuda()

pred = torch.FloatTensor()
pred = pred.cuda()

for batch_x, batch_y in test_loader:
    gt = torch.cat((gt, batch_y.cuda()), 0)
    batch_x = Variable(batch_x.cuda(), volatile=True)
    batch_y = Variable(batch_y.cuda(), volatile=True)
    num_batch+=1

    out = model(batch_x)

    pred = torch.cat((pred, out.data), 0)

val_AUROCs = compute_AUCs(gt, pred,nnClassCount)
val_AUROC_avg = np.array(val_AUROCs).mean()
print('The train average AUROC is: {:.8f},'.format(val_AUROC_avg))
for i in range(nnClassCount):
    print('The test_AUROC of {} is {}'.format(CLASS_NAMES[i], val_AUROCs[i]))






