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
from utils.loss import cross_entropy2d

gpus = [0,1]
#gpus = [0]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_name = 'fcn'
encoder_name = 'resnet101'

model = get_encoder_decoder(encoder_name, decoder_name, num_classes=34, fpn=True)
model = model.to(device)

if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
target_transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor()])
dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)
dataiter = iter(data_loader)
data,label = dataiter.next()
label = label*255
#print(np.unique(label.numpy()))
imshow(torchvision.utils.make_grid(data))

#print(data.size())
#print(label.size())
#print(label.long())
#print(model)
t = time.time()
for i in range(1000):
    data,label = dataiter.next()
    label = label*255
    label = torch.squeeze(label.permute(0,2,3,1))
    model.zero_grad()
    output = model(data.to(device))
    om = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()
    
    loss = cross_entropy2d(output, label.long().to(device))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(i,loss.item())
    
    loss.backward()
    optimizer.step()
elapsed = time.time() - t


print(elapsed/100)

def decode_segmap(image, nc=33):
   
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               (0, 64, 0), (128, 64, 0)])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(om[0])
plt.imshow(rgb)