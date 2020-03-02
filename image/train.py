# train.py 将用数据集训练新的网络，并将模型保存为检查点。 
#基本用途：python train.py data_directory
# 在训练网络时，输出训练损失、验证损失和验证准确率
# 选项：
# 设置保存检查点的目录：python train.py data_dir --save_dir  save_directory
# 选择架构：python train.py data_dir --arch "vgg13"
# 设置超参数：python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# 使用 GPU 进行训练：python train.py data_dir --gpu
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


import time
from torch.autograd import Variable

import json
from collections import OrderedDict
from PIL import Image
import seaborn as sb

def dataloaders():
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloaders = torch.utils.data.DataLoader(validation_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=20, shuffle=True)
    return trainloaders,validationloaders,testloaders,train_datasets

def jsonLoad(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def setModelParams(model, learning_rate):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 5000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(5000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model,criterion,optimizer
    
def do_deep_learning(model, dataloaders_map, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        for phase in dataloaders_map:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0
            steps = 0
            for ii, (inputs, labels) in enumerate(dataloaders_map[phase]):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):   
                    # Forward and backward passes
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    #backward和梯度更新只在训练时候使用
                    if phase =='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
            
                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))

                    running_loss = 0   
                    
            
def check_accuracy_on_test(testloader,model):    
    correct = 0
    total = 0
     # change to cuda
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
def save_check_point(model,check_point_file,train_datasets):
    checkpoint = {'classifier':model.classifier,
                  'category':train_datasets.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, check_point_file)
   