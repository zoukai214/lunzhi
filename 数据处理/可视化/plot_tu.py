import numpy as np
import pandas as pd

all_data = pd.read_csv('result.txt',header = None,sep= ' ')
#这个位置略显蠢，应该用个循环
a = all_data[all_data.iloc[:,1]==1]
b = all_data[all_data.iloc[:,2]==1]
c = all_data[all_data.iloc[:,3]==1]
d = all_data[all_data.iloc[:,4]==1]
e = all_data[all_data.iloc[:,5]==1]
f = all_data[all_data.iloc[:,6]==1]
g = all_data[all_data.iloc[:,7]==1]
h = all_data[all_data.iloc[:,8]==1]
i = all_data[all_data.iloc[:,9]==1]
j = all_data[all_data.iloc[:,10]==1]
k = all_data[all_data.iloc[:,11]==1]
l = all_data[all_data.iloc[:,12]==1]
m = all_data[all_data.iloc[:,13]==1]
n = all_data[all_data.iloc[:,14]==1]
#print(a.head(10))
print(a.shape[0])
print(b.shape[0])
print(c.shape[0])
print(d.shape[0])
print(e.shape[0])
print(f.shape[0])
print(g.shape[0])
print(h.shape[0])
print(i.shape[0])
print(j.shape[0])
print(k.shape[0])
print(l.shape[0])
print(m.shape[0])
print(n.shape[0])
a,b,c,d,e,f,g,h,i,j,k,l,m,n = a.shape[0],b.shape[0],c.shape[0],d.shape[0],e.shape[0],f.shape[0],g.shape[0],h.shape[0],i.shape[0],j.shape[0],k.shape[0],l.shape[0],m.shape[0],n.shape[0]


# -*- coding: utf-8 -*-
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.figure(figsize=(18,6))
# 构建数据
number1 = [a,b,c,d,e,f,j,h,i,j,k,l,m,n]
print(number1)
number2 = number1.sort(reverse=True)
print(number2)
labs = [0.826,0.909,0.885,0.713,0.864,0.804,0.768,0.882,0.814,0.900,0.935,0.853,0.789,0.925]
#plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
#plt.rcParams['axes.unicode_minus'] = False
# 绘图
plt.bar(range(14), number1, align = 'center',color='steelblue', alpha = 0.8)
# 添加轴标签
plt.ylabel('the number of patients')
# 添加标题
plt.title('ChestX14 dataset ')
# 添加刻度标签
#plt.xticks(range(14),[ u'肺不张', u'心肥大', '积液', '渗透', '肺肿块', '肺结节', '肺炎',
               # '气胸', '硬化', '肺水肿', '肺气肿', '肺纤维化', '胸膜增厚', '疝气'])
plt.xticks(range(14), ['渗透','积液','肺不张', '肺结节','肺肿块','气胸','硬化', '胸膜增厚',u'心肥大','肺气肿','肺炎','肺水肿', '肺纤维化','疝气'],fontproperties=font)
# 设置Y轴的刻度范围
plt.ylim([0,23000])

# 为每个条形图添加数值标签
for x,y in enumerate(number1):
    plt.text(x,y+100,'%s' %round(y,1),ha='center')# 显示图形plt.show()
#for x,y in enumerate(labs):
    #plt.text(x,GDP[x]+800,'{}'.format(y),ha = 'center')
plt.show()