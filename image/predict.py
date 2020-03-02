# predict.py 将使用训练的网络预测输入图像的类别。
# 基本用途：python predict.py input checkpoint
# 选项：
# 返回前 KK 个类别：python predict.py input checkpoint --top_k 3
# 使用类别到真实名称的映射：python predict.py input checkpoint --category_names cat_to_name.json
# 使用 GPU 进行训练：python predict.py input checkpoint --gpu
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

def load_checkpoint(filepath, model, train_datasets):
    checkpoint = torch.load(filepath)
    
    model.classifier = checkpoint['classifier']
    train_datasets.class_to_idx = checkpoint['category']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model,train_datasets

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #     im = Image.open(image)
    #     np_image = np.array(im)
    image_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    im = Image.open(image)
    img_tensor = image_transforms(im)
    return img_tensor


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)