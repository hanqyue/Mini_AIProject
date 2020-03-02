import argparse
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
# Imports print functions that check the lab
import train 
import predict

def setModel():
    model = models.vgg16(pretrained=True)
      # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model
            
def main():
    in_args =get_input_args()
    #加载数据
    trainloaders,validationloaders,testloaders,train_datasets = train.dataloaders()
    cat_to_name = train.jsonLoad(in_args.category_names)
    #构建网络
    model = setModel()
    model, criterion, optimizer= train.setModelParams(model,in_args.learning_rate)
    #训练
    dataloaders_map = {"train": trainloaders, "validation": validationloaders}
    train.do_deep_learning(model, dataloaders_map, in_args.epochs, 40, criterion, optimizer, in_args.gpu)
    #测试：
    train.check_accuracy_on_test(testloaders,model)
    #保存
    train.save_check_point(model,in_args.save_dir,train_datasets)
    #加载
    data_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    train_dir = 'train'
    dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    model,dataset = predict.load_checkpoint(in_args.save_dir, model, dataset)
    #预测
    probs, classes = predict.predict("train/10/image_07087.jpg", model, in_args.top_k)
    print(probs)
    print(classes)
    
   
def get_input_args():
    parser = argparse.ArgumentParser()
    # 设置保存检查点的目录：python train.py data_dir --save_dir  save_directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = '保存检查点文件') 
    parser.add_argument('--arch', type = str, default = 'vgg16', help = '选择架构') 
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = '学习速率')
    parser.add_argument('--hidden_units', type = int, default = 512, help = '隐藏点数量')
    parser.add_argument('--epochs', type = int, default = 20, help = '训练次数') 
    parser.add_argument('--gpu', type = str, default = 'gpu', help = '使用 GPU 进行训练') 
    
    parser.add_argument('--top_k', type = int, default = 3, help = '返回前 KK 个类别') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = '类别到真实名称的映射文件')

    in_args = parser.parse_args()
    return in_args
    
if __name__ == "__main__":
    main()