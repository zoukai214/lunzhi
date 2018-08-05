import torch.nn.functional as F
import torch
def BCEloss_weight(inputs, target):
    #inputs=torch.from_numpy(inputs)
    #target=torch.from_numpy(target)
    #idx = torch.randperm(target.shape[0]) 
    #idx = Variable(idx).cuda()

    probability = F.softmax(inputs) #输出概率
    #新建dev
    target=target.float()

    #probability=torch.index_select(probability,1,idx[1])
    probability=probability[:,1] #预测为正类的概率
    #print(torch.log(probability))
    add_one=torch.mul(target,torch.log(probability))
    #print(probability)
    #add_one=target+torch.log(probability)
    add_two=torch.mul((1-target),torch.log(1-probability))

    loss=add_one+add_two
    loss=torch.sum(loss)
    return loss