from PIL import Image
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import random
from resnet18 import ResNet
# https://github.com/li554/resnet18-cifar10-classification
# https://github.com/ZOMIN28/ResNet18_Cifar10_95.46/blob/main/utils/readData.py
# https://github.com/ZQPei/transfer_learning_resnet18/blob/master/transfer_learning_finetuning.py
# https://github.com/Mountchicken/ResNet18-CIFAR10/tree/main

# pytorch关于heatmap和层Frozen
# https://pytorch.org/tutorials/beginner/introyt/captumyt.html
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

# 分割数据 将50000个训练图像留出20%作训练中的验证集，具体是50000个id先打乱重拍再拆分
valratio = 0.2
train_num =50000
split = int(train_num*(1-valratio))
total_ids = list(range(train_num))
random.shuffle(total_ids)
train_ids = total_ids[:split]
val_ids   = total_ids[split:]
# 再封装到batch之前，必须至少先转换为tensor数据。注意：变换后的图像是在batch中，不会改变原始数据
transform = {'train':transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()
                                         ,transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
              'test':transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])}
train_data = datasets.CIFAR10('.',train=True,transform=transform['train'],download=True)
val_data = datasets.CIFAR10('.',train=True,transform=transform['test'],download=True)
test_data = datasets.CIFAR10('.',train=False,transform=transform['test'],download=True)
#batch不足128时，不舍弃
train_loader = DataLoader(train_data,128,sampler=SubsetRandomSampler(train_ids))
val_loader = DataLoader(val_data,128,sampler=SubsetRandomSampler(val_ids))
test_loader = DataLoader(test_data,128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'cur device is {device}')
model = ResNet()
# cifar图像大小为3x32x32，原7x7卷积，最后一层pooling的输入维度是512x1x1，改用3x3卷积，最后一层pooling的输入维度是512x2x2
model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
model.to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

epoch = 100
count = 1
lr = 0.1
accuracy_max = 0
for i in range(1,epoch+1):
#整个训练集迭代一次，就验证一次
#每在训练集迭代一次，学习率都可能进行调整，因此每迭代一次要重新创建优化器
    # 调整学习率
    if count%10==0:
        count=0
        lr = lr*0.5
    #优化器管理模型参数，当调用backward时会计算和记录参数梯度，优化器调用step时会根据当前参数梯度计算参数的更新 
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    model.train()
    total_loss = 0.0
    for batch,labels in train_loader:
        batch = batch.to(device)
        labels = labels.to(device)
        # 五步：前向，计损，反传，更新，清零
        out = model(batch).to(device)
        loss = criterion(out,labels)
        # loss是一个标量的tensor，需要调用item()
        total_loss +=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'epoch:{i} | average train loss:{total_loss/40000}')    
    model.eval()
    val_loss = 0.0
    correct = 0
    # 验证时关闭梯度计算，可以提高计算速度，验证关注的是准确率，训练关注的是损失
    with torch.no_grad():
        for batch,labels in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)  
            out = model(batch).to(device)
            loss = criterion(out,labels)
            val_loss += loss.item()
            pred = torch.argmax(out,dim=1)
            correct += (pred==labels).sum().item()
    cur_accuracy = correct/10000
    print(f'epoch:{i} | average val loss:{val_loss/10000} | average precision:{cur_accuracy}') 
    print(f'epoch:{i} cur learning rate is {lr}')
    if cur_accuracy>accuracy_max:
        accuracy_max = cur_accuracy
        torch.save(model.state_dict(),'checkpoint/epoch{:0>3}resnet18_cifar10.pt'.format(i))
    count = count+1
torch.save(model.state_dict(),'checkpoint/last_resnet18_cifar10.pt')
