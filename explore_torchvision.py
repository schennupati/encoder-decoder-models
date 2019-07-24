#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.datasets import Cityscapes
from torch import nn
from torchvision import transforms

from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, transform_targets,convert_targets

#gpus = [0,1]
gpus = list(range(torch.cuda.device_count()))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_name = 'fcn'
encoder_name = 'resnet50'
im_size = 256 
n_classes = 19



transform = transforms.Compose([transforms.Resize(im_size),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

target_transform = transforms.Compose([transforms.Resize(im_size),transforms.ToTensor()])

train_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)
val_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='val', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)

dataiter = iter(train_loader)
data,targets = dataiter.next()
targets = convert_targets(transform_targets(targets))

#print(np.unique(targets[0]))
#rgb = decode_segmap(targets[0])
#plt.imshow(rgb)


#print(np.unique(label.numpy()))
#imshow(torchvision.utils.make_grid(data))

#print(data.size())
#print(label.size())
#print(label.long())
#print(model)

model = get_encoder_decoder(encoder_name, decoder_name, num_classes=n_classes, fpn=True)
model = model.to(device)
if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

running_metrics_val = runningScore(n_classes)

train_loss_meter = averageMeter()
val_loss_meter   = averageMeter()
time_meter       = averageMeter()
    
optimizer = optim.Adam(model.parameters(), lr=0.0001)
  
for epoch in range(5):
    print('********************** '+str(epoch+1)+' **********************')
    for i, data in enumerate(train_loader, 0):
        t = time.time()
        
        inputs,targets = data
        targets = convert_targets(transform_targets(targets))
        #labels = labels*255
        #labels = torch.squeeze(labels.permute(0,2,3,1))
                
        optimizer.zero_grad()
        
        outputs = model(inputs.to(device))
        loss = cross_entropy2d(outputs, targets.long().to(device))
        loss.backward()
        optimizer.step()
        
        time_meter.update(time.time() - t)
        
        train_loss_meter.update(loss.item())
        
        if i % 10 == 9:        
            print('epoch: %d batch: %5d time_per_batch: %.3f  loss: %.3f' %
                      (epoch + 1, i + 1, time_meter.avg , train_loss_meter.avg))
            running_loss = 0.0
            train_loss_meter.reset()
            time_meter.reset()
        
    with torch.no_grad():
        for data in val_loader:
            images, targets = data
            targets = convert_targets(transform_targets(targets))
            #labels = labels*255
            #labels = torch.squeeze(labels.permute(0,2,3,1))
            outputs  = model(images.to(device))
            val_loss = cross_entropy2d(outputs, targets.long().to(device))
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = targets.data.cpu().numpy()
            
            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())
            
    score, class_iou = running_metrics_val.get_scores()
            
    for k,v in score.items():
        print(k,v)
                
    for k,v in class_iou.items():
        print(k,v)
            
    running_metrics_val.reset()
    val_loss_meter.reset()
                        
om = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
rgb = decode_segmap(om[0])
plt.imshow(rgb)
