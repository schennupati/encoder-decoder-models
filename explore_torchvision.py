#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import torch
import torchvision
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import Cityscapes
from torch import nn
from models.encoder_decoder import get_encoder_decoder
from torchvision import transforms

gpus = [0,1]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_name = 'fcn'
encoder_name = 'resnet101'

model = get_encoder_decoder(encoder_name, decoder_name, num_classes=33, fpn=True)
model = model.to(device)

if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
target_transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)
dataiter = iter(data_loader)
data,label = dataiter.next()

imshow(torchvision.utils.make_grid(data))
criterion = nn.CrossEntropyLoss()

print(data.size())
print(label.size())
#print(model)
t = time.time()
for i in range(100):
    data,label = dataiter.next()
    output = model(data.to(device))
    loss = criterion(output, label.long().to(device))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(i,loss)
    model.zero_grad()
    loss.backward()
    optimizer.step()
elapsed = time.time() - t

print(elapsed/100)