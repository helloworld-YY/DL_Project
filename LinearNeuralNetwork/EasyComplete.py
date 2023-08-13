import torch
import random

from torch.utils import data
from torch import nn


def CreatData(w, b, num_examples):
    
    X = torch.normal(0, 1, (num_examples, len(w))) #矩阵乘法
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #加入噪声
    
    return X, y.reshape(-1,1) #y转化为列向量

def ExactBatchData(data_arrays, batch_size, is_train = True):  #返回是一个Torch迭代器，可以不断取值
    
    dataset = data.TensorDataset(*data_arrays)  #*表示可以传入多个参数
    
    return data.DataLoader(dataset, batch_size, shuffle= is_train)


def NNImplementation():
    num_examples = 1000
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = CreatData(true_w, true_b, num_examples)
    batch_size = 10
    
    data_iter = ExactBatchData((features, labels), batch_size)
    
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
    
    num_epochs = 5

    
    for epoch in range(num_epochs):  #每一次迭代
        for X, y in data_iter:  #for可以访问可迭代对象
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss {l:f}')
    w = net[0].weight.data
    print('w的估计误差: ', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差: ', true_b - b)

if __name__ == "__main__":
    NNImplementation()