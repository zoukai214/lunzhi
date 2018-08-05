def computeAUROC (dataGT, dataPRED, classCount):
        thre = []
        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        np.savetxt('/home/kenneth/zoukai/f1_output/crop_gt.csv',datanpGT,delimiter=',',fmt = '%5.2f')
        np.savetxt('/home/kenneth/zoukai/f1_output/crop_pre.csv',datanpPRED,delimiter = ',',fmt='%5.4f')
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            fpr, tpr, thresholds = roc_curve(datanpGT[:, i], datanpPRED[:, i])
			#敏感性：tpr   特异性：1-fpr
            c = tpr + 1 - fpr
            c = c.tolist()
            id = c.index(max(c))
            thre.append(thresholds[id])
        print(thre)
        return outAUROC
