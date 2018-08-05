import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#计算各类指标
def computef1(dataGT, dataPRED, flodthord):
    class_ff = []
    c = np.where(dataPRED > flodthord, 1, 0)
    #也可以选择top3进行计算
    for j in range(dataGT.shape[0]):
        a = data[j]
        b = np.argpartition(a, -3)[-3:]
        activate_classes = b[np.where(a[b] > 0)]
        # print(len(activate_classes))
        #if len(activate_classes) == 0:
        #    data[j] = 0
        #else:
        #    c = min(a[activate_classes])
        #    c = np.where(a >= c, 1, 0)
        #    data[j] = c

    for i in range(len(flodthord)):
        class_tp = []
        positive = np.sum(dataGT[:, i] == 1)
        negative = np.sum(dataGT[:, i] == 0)
        TP = int(np.sum(np.multiply(dataGT[:, i], c[:, i])))
        FP = np.sum(np.logical_and(np.equal(dataGT[:, i], 0), np.equal(c[:, i], 1)))
        FN = np.sum(np.logical_and(np.equal(dataGT[:, i], 1), np.equal(c[:, i], 0)))
        TN = np.sum(np.logical_and(np.equal(dataGT[:, i], 0), np.equal(c[:, i], 0)))

        acc = accuracy_score(dataGT[:, i], c[:, i])
        p = precision_score(dataGT[:, i], c[:, i], average='binary')
        r = recall_score(dataGT[:, i], c[:, i], average='binary')
        f1score = f1_score(dataGT[:, i], c[:, i], average='binary')

        class_tp.append(positive)
        class_tp.append(negative)
        class_tp.append(TP)
        class_tp.append(FP)
        class_tp.append(FN)
        class_tp.append(TN)
        class_tp.append(acc)
        class_tp.append(p)
        class_tp.append(r)
        class_tp.append(f1score)
        class_ff.append(class_tp)
    return class_ff


def choicef1(dataGT, dataPRED, flodthord, bg, end, number, sort):
    f1 = []
    dia = []
    cs = np.linspace(bg, end, number)
    for i in range(len(cs)):
        flodthord[sort] = cs[i]
        c = np.where(dataPRED > flodthord, 1, 0)
        f1score = f1_score(dataGT[:, sort], c[:, sort], average='binary')
        f1.append(f1score)
    id = x.index(max(x))
    print("f1:", x[id])
    print("阈值：", cs[id])
    # return f1, cs


def computefload(dataGT, dataPRED, classCount):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    # np.savetxt('./output/val_pre.csv',datanpPRED,delimiter = ',',fmt='%5.4f')
    for i in range(classCount):
        fpr, tpr, thresholds = roc_curve(datanpGT[:, i], datanpPRED[:, i])
        c = tpr + 1 - fpr
        # print(c.shape)
        c = c.tolist()
        id = c.index(max(c))
        outAUROC.append(thresholds[id])
    return flodthord


dataGT = np.loadtxt('./output/val_gt.csv', delimiter=',', dtype='float')
dataPRED = np.loadtxt('./output/val_pre.csv', delimiter=',', dtype='float')
flodthord = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
zz = computef1(dataGT, dataPRED, flodthord)

#保存为csv文件
#label = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
#columns = ['positive','negative','TP','FP','FN','TN','acc','precision','recall','f1']
#df = pd.DataFrame()
#for k in range(0,len(zz)):
#    for j in range(0,len(zz[0])):
 #       df.iloc[k,j]=zz[k][j]
#dff = df.T
#dff.to_csv('12.csv',index=label, header=columns,sep = ' ')