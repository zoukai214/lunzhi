# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from PIL  import Image
from dry_oily import face_markpoints,face_crop
#import home.ubuntu.skin_demo.dry_oily.face_crop

import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import random

img_dir = "../data/output/"
txt_dir = "../"
model_save_dir = "../models/model_four_crop_pyTorch/"


def predict(img,model):
    #img = Image.open('../2016080809423368148.jpg')
    img = face_crop.crop(img, request_region = ['left_cheek', 'right_cheek', 'chin', 'nose'])
    img = Image.fromarray(img).convert('RGB')
    #cur_time = time.time()
    img = img.resize((224, 224))
    #print('time passed', time.time() - cur_time)
    model = model.cuda()
    #img = img.cuda()
    #print(model)

    #img = cv2.resize(img,(224, 224))
    #img = np.array(img)

    #print(img[img != 0])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    model.eval()
    img_tensor = transform(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor.cuda())
    #print(img_variable)
    #cur_time = time.time()
    y_predict = model(img_variable)
    print("The Predict Result is : ",y_predict)
    out = torch.max(y_predict,1)[1]
    #print('time model predict', time.time() - cur_time)
    return out
    #print("more max is : ",torch.max(y_predict, 1))[1]
    #print(y_predict.shape)
    #print("The Predict Result is : ", y_predict)


#img = cv2.imread('../2016080809423368148.jpg')
#

model = torch.load(model_save_dir+'model_17_0.045009_0.797938.pkl')
#for i in xrange(8):
#filename = '../'+str(i+1)+'.jpg'
img = cv2.imread('../2.jpg')
#img = np.array(img)
#img = Image.fromarray(img).convert('RGB')
#cur_time = time.time()
#img = img.resize((224, 224))
#print('time passed', time.time() - cur_time)
#model = model.cuda()
#img = img.cuda()
#print(model)

#img = cv2.resize(img,(224, 224))
#img = np.array(img)

#print(img[img != 0])

#transform = transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#model.eval()
#img_tensor = transform(img)
#img_tensor.unsqueeze_(0)
#img_variable = Variable(img_tensor.cuda())
#print(img_variable)
#cur_time = time.time()
#y_predict = model(img_variable)
#print("The Predict Result is : ",y_predict)
#out = torch.max(y_predict,1)[1]
#print('time model predict', time.time() - cur_time)
#return out
out = predict(img,model)
print(out)
#
# print("max is : ", out)
# print('time passed',time.time()-cur_time)

