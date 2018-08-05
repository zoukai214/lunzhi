import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

test_txt_path = '/home/ubuntu/skin_demo/ChestXray/CheXNet-with-localization-master/dataset/tmp.txt'
img_folder_path = '/home/ubuntu/yangwenbin/chexnet-master/database/'

with open(test_txt_path, "r") as f:
    test_list = [i.strip() for i in f.readlines()]

print("number of test examples:",len(test_list))

test_X = []
print("load and transform image")
for i in range(len(test_list)):
    image_path = os.path.join(img_folder_path, test_list[i])
    img = scipy.misc.imread(image_path)
    if img.shape != (1024,1024):
        img = img[:,:,0]
    img_resized = skimage.transform.resize(img,(256,256))
    test_X.append((np.array(img_resized)).reshape(256,256,1))
    if i % 100==0:
        print(i)
test_X = np.array(test_X)
print('len(test_X)',len(test_X))

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

pathModel = '/home/ubuntu/skin_demo/ChestXray/master/models/m-25012018-123527.pth.tar'
model = DenseNet121(14, True).cuda()
model = torch.nn.DataParallel(model).cuda()
modelCheckpoint = torch.load(pathModel)
model.load_state_dict(modelCheckpoint['state_dict'])
print("model loaded")

class ChestXrayDataSet_plot(Dataset):
	def __init__(self, input_X = test_X, transform=None):
		self.X = np.uint8(test_X*255)
		self.transform = transform

	def __getitem__(self, index):
		"""
		Args:
		    index: the index of item
		Returns:
		    image
		"""
		current_X = np.tile(self.X[index],3)
		image = self.transform(current_X)
		return image
	def __len__(self):
		return len(self.X)

test_dataset = ChestXrayDataSet_plot(input_X = test_X,transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))

# thresholds = np.load("thresholds.npy")
thresholds = np.array([0.19362465, 0.07700258, 0.3401143 , 0.39875817, 0.08521137,
       0.14014415, 0.02213187, 0.08226113, 0.3401143 , 0.39875817, 0.08521137,
       0.14014415, 0.02213187, 0.08226113],dtype='float32')
print("activate threshold",thresholds)

class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
#         self.probs = F.softmax(self.preds)[0]
#         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


heatmap_output = []
image_id = []
output_class = []

gcam = GradCAM(model=model, cuda=True)

for index in range(len(test_dataset)):
    input_img = Variable((test_dataset[index]).unsqueeze(0).cuda(), requires_grad=True)
    probs = gcam.forward(input_img)

    activate_classes = np.where((probs > thresholds)[0]==True)[0] # get the activated class
    for activate_class in activate_classes:
        gcam.backward(idx=activate_class)
        output = gcam.generate(target_layer="module.densenet121.features.denseblock4.denselayer16.conv2")
        #### this output is heatmap ####
        if np.sum(np.isnan(output)) > 0:
            print("fxxx nan")
        heatmap_output.append(output)
        image_id.append(index)
        output_class.append(activate_class)
    print("test ",str(index)," finished")

print("heatmap output done")
print("total number of heatmap: ",len(heatmap_output))
print('image_id',image_id)
print('output_class',output_class)


pathImageFile = img_folder_path + 'images_004/00008473_002.png'
raw_image = cv2.imread(pathImageFile, 1)
raw_image = cv2.resize(raw_image, (224, 224))
filename = 'test.jpg'
def feature_save(filename, gcam1,gcam2,cls1,cls2, raw_image):
        gcamall = np.add(gcam1, gcam2)
        cam = gcam1 / np.max(gcam1)
        mask = np.where(cam>0.8)
        x0 = mask[0].min()
        x1 = mask[0].max()
        y0 = mask[1].min()
        y1 = mask[1].max()

        cam = gcam2/ np.max(gcam2)
        mask = np.where(cam>0.8)
        x0_2 = mask[0].min()
        x1_2 = mask[0].max()
        y0_2 = mask[1].min()
        y1_2 = mask[1].max()

        gcamall = cv2.applyColorMap(np.uint8(gcamall * 255.0), cv2.COLORMAP_JET)
        gcamall = gcamall.astype(np.float)*0.5 + raw_image.astype(np.float)
        gcamall = gcamall / gcamall.max() * 255.0
        cv2.rectangle(gcamall,(y0,x0),(y1,x1),(55,255,155),2)
        cv2.putText(gcamall, cls1, (y0,x0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.rectangle(gcamall,(y0_2,x0_2),(y1_2,x1_2),(55,255,155),2)
        cv2.putText(gcamall, cls2, (y0_2,x0_2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(filename, np.uint8(gcamall))
class_index = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
feature_save(filename, heatmap_output[5], heatmap_output[6],class_index[output_class[5]],\
             class_index[output_class[6]], raw_image)
