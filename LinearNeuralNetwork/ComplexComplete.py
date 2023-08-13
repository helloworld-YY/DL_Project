import torch
import random

from torch.utils import data
from torch import nn

'''
通过梯度下降优化求解线性回归问题
'''

def CreatData(w, b, num_examples):
    
    X = torch.normal(0, 1, (num_examples, len(w))) #矩阵乘法
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #加入噪声
    
    return X, y.reshape(-1,1) #y转化为列向量

def LinearReg(X, w, b):
    
    return torch.matmul(X, w) + b

def MSE(y_pred, y):
    
    return (y_pred - y.reshape(y_pred.shape))**2 / 2 #此时还未求和

def SGD(paramenters, alpha, batch_size):
    with torch.no_grad():
        for param in paramenters:
            param -= alpha*param.grad / batch_size
            param.grad.zero_()
            
            
def ExactBatchData(batch_size, features, labels):
    num_examples = len(features)
    index = list(range(num_examples)) #获取样本数据的索引
    random.shuffle(index)  #打乱是顺序
    
    for i in range(0, num_examples, batch_size):
        batch_index = torch.tensor(index[i:min(i + batch_size, num_examples)])
        yield features[batch_index, :], labels[batch_index] #返回一次抽取结果

def ComplexImplementation():
    alpha = 0.01
    num_epochs = 5
    net = LinearReg
    loss = MSE
    num_examples = 1000
    batch_size = 10
    
    true_w = torch.tensor([2, -3.4])
    ture_b = 4.2
    
    w = torch.normal(0, 1, size = (2, 1), requires_grad= True)
    b = torch.zeros(1, requires_grad= True)
    
    features, labels = CreatData(true_w, ture_b, num_examples)
    
    for epoch in range(num_epochs):  #每一次迭代
        for X, y in ExactBatchData(batch_size, features, labels):
            
            l = loss(LinearReg(X, w, b), y)
            l.sum().backward()
            SGD([w, b], alpha, batch_size)
            
        with torch.no_grad():
            CurrentLoss = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss:{float(CurrentLoss.mean()):f}')
        
    print("w的误差为:",(w - true_w.reshape(w.shape)),"\nb的误差为: ", b- ture_b)





        
if __name__ == "__main__":
    ComplexImplementation()