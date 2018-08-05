import numpy as np
import pandas as pd
from sklearn.utils import shuffle
#CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
all_data = pd.read_csv('result.txt',header = None,sep= ' ')
#去掉不需要的类别
pneu = all_data.drop([1,2,3,4,5,6,8,9,10,11,12,13,14],axis=1)

#取正例与负例
pneu_data=pneu[pneu.iloc[:,1]==1]
pneu_zero_data = pneu[pneu.iloc[:,1]==0]
#选出一定量的负例并组合
pneu_concat = pd.concat([pneu_data,pneu_zero_data.iloc[:6764,:]],axis=0,ignore_index=True)
df = shuffle(pneu_concat)
#划分数据集
penu_train = df.iloc[:5682,:]
penu_val = df.iloc[5682:7205,:]
penu_test = df.iloc[7205:,:]
penu_test.to_csv('C:\\Users\\Administrator\\Desktop\\数据集\\肺炎数据集\\penu_test_1.txt',header=None,sep=' ',index=None)
penu_val.to_csv('C:\\Users\\Administrator\\Desktop\\数据集\\肺炎数据集\\penu_val_1.txt',header=None,sep=' ',index=None)
penu_train.to_csv('C:\\Users\\Administrator\\Desktop\\数据集\\肺炎数据集\\penu_train_1.txt',header=None,sep=' ',index=None)