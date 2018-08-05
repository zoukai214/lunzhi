from __future__ import division,print_function
import os
import sys
sys.path.append('/home/ubuntu/kai/DR/gdcm-build/bin')
import gdcm
import pydicom as dicom

from PIL import Image
import pylab
import pylab
def process_dcm(filepath):
    file_name = os.path.basename(filepath)
    filename, file_extension = os.path.splitext(file_name)
    ds = dicom.read_file(filepath)
    # print(ds)
    img = ds.pixel_array

    print(img[150:160,150:160])
    # img = 3000 - img

    # img=np.transpose(img,[1,0])
    # img = Image.open(filename + '.png').convert('RGB')
    # img = Image.open(filename + '.png')
    # print(img.dtype)
    # print(img.shape)
    # pylab.imsave(filename + '.jpg', img)
    pylab.imsave(filename+'.jpg', img, cmap=pylab.cm.gray)
    #img = Image.open(filename+'.jpg').convert('RGB')


    # cv2.imwrite(filename+'.png',img,[cv2.IMWRITE_PNG_COMPRESSION, 9])
    # img = Image.open(filename+'.png').convert('RGB')
    # os.remove(filename+'.png')
    #return img
filepath = '/home/ubuntu/skin_demo/ChestXray/CheXNet-with-localization-master/dicom_test/disease_check/yiwancheng/feibuzhang/600729286_20180313_1_1.dcm'
process_dcm(filepath)