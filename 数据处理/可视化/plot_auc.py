import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

#dataGT = np.loadtxt(r'F:\yiyuan\lunzi\data\ban_gt.csv', delimiter=',', dtype='float')
#dataPRED = np.loadtxt(r'F:\yiyuan\lunzi\data\ban_pre.csv', delimiter=',', dtype='float')

dataGT = np.loadtxt(r'C:\Users\Administrator\Desktop\数据集\cls6_val11_gt.csv', delimiter=',', dtype='float')
dataPRED = np.loadtxt(r'C:\Users\Administrator\Desktop\数据集\cls6_val11_pre.csv', delimiter=',', dtype='float')

plt.figure(figsize=(18,6))
fpr1,tpr1,thresholds = roc_curve(dataGT[:,0],dataPRED[:,0])
fpr2,tpr2,thresholds = roc_curve(dataGT[:,1],dataPRED[:,1])
fpr3,tpr3,thresholds = roc_curve(dataGT[:,2],dataPRED[:,2])
fpr4,tpr4,thresholds = roc_curve(dataGT[:,3],dataPRED[:,3])
fpr5,tpr5,thresholds = roc_curve(dataGT[:,4],dataPRED[:,4])
fpr6,tpr6,thresholds = roc_curve(dataGT[:,5],dataPRED[:,5])

plt.plot(fpr1,tpr1,linewidth=2,label="肺不张")
plt.plot(fpr2,tpr2,linewidth=2,label="积液")
plt.plot(fpr3,tpr3,linewidth=2,label="渗透")
plt.plot(fpr4,tpr4,linewidth=2,label="肺肿块")
plt.plot(fpr5,tpr5,linewidth=2,label="肺结节")
plt.plot(fpr6,tpr6,linewidth=2,label="气胸")

plt.xlabel("false presitive rate",fontsize=18)

plt.ylabel("true presitive rate",fontsize=18)

plt.ylim(0,1.05)

plt.xlim(0,1.05)

plt.legend(loc=4,prop={ 'weight':'normal', 'size'  :23,  }  )#图例的位置
#plt.legend(loc=4, numpoints=2)
#leg = plt.gca().get_legend()
#ltext  = leg.get_texts()
#plt.setp(ltext, fontsize='normal')
plt.savefig("examples.jpg")
plt.show()