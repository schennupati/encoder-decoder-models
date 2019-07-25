#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import torch
import time
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import glob

from torchvision.datasets import Cityscapes
from torch import nn
from torchvision import transforms

from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, transform_targets, convert_targets, cat_labels
#from utils.im_utils import get_class_weights

gpus = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_name = 'fcn'
encoder_name = 'resnet18'
im_size = 256
batch_size = 4 
n_classes = 19
pyramid_networks = False
best_iou = -100.0
resume_training = True

base_dir =  os.path.join(os.path.expanduser('~'),'results')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

decoder = 'fpn' if pyramid_networks else 'fcn'    
exp_name  =  (encoder_name + '-' + str(decoder) +
             '-' + str(im_size) + '-' + str(n_classes))
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

transform = transforms.Compose([transforms.Resize(im_size),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

target_transform = transforms.Compose([transforms.Resize(im_size),transforms.ToTensor()])

train_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)
val_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='val', mode='fine',
                     target_type='semantic',transform=transform,target_transform=target_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

dataiter = iter(train_loader)
data,targets = dataiter.next()
targets = convert_targets(transform_targets(targets))

model = get_encoder_decoder(encoder_name, decoder_name, num_classes=n_classes, fpn=pyramid_networks)
model = model.to(device)

if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

running_metrics_val = runningScore(n_classes)

train_loss_meter = averageMeter()
val_loss_meter   = averageMeter()
time_meter       = averageMeter()
    
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#class_weights = get_class_weights(train_loader,n_classes)
class_weights = [3.0309219731075485, 12.783811398373516, 4.6281263808532, 
                 33.466829282994446, 32.37214980477001, 33.69941903632168, 
                 41.312452367374135, 35.31661830466744, 6.191211436807339, 
                 30.51037629066412, 17.300065531616138, 32.314456409694635, 
                 44.92898754596857, 11.933240015436805, 44.386160408368625, 
                 45.16117055995425, 45.113897310212835, 48.01339129145941, 
                 43.18769924298604] #CityScapes 19 Classes weights

class_weights = torch.FloatTensor(class_weights).cuda()

list_of_models = glob.glob(os.path.join(base_dir,'*.pkl'))

if any(exp_name in model for model in list_of_models) and resume_training:
    latest_model = max(list_of_models, key=os.path.getctime)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_iter = int(checkpoint["epoch"]/len(train_loader))-1
    print("Loaded checkpoint '{}' (epoch {})".format(latest_model, start_iter))
    
else:
    print("Begining Training from Scratch")

for epoch in range(1):
    print('********************** '+str(epoch+1)+' **********************')
    for i, data in enumerate(train_loader, 0):
        t = time.time()
        
        inputs,targets = data
        targets = convert_targets(transform_targets(targets))
                
        optimizer.zero_grad()
        
        outputs = model(inputs.to(device))
        loss = cross_entropy2d(outputs, targets.long().to(device),weight=class_weights)
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
            outputs  = model(images.to(device))
            val_loss = cross_entropy2d(outputs, targets.long().to(device),weight=class_weights)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = targets.data.cpu().numpy()
            
            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())
            
    score, class_iou = running_metrics_val.get_scores()
            
    for k,v in score.items():
        print(k,v)
                
    for k,v in class_iou.items():
        print(cat_labels[k].name,': ',v)
            
    running_metrics_val.reset()
    val_loss_meter.reset()
    
    if score["Mean IoU : \t"] >= best_iou:
        best_iou = score["Mean IoU : \t"]
        state = {"epoch": i + 1,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "best_iou": best_iou}
        save_path = os.path.join(base_dir,"{}_{}_best_model.pkl".format(exp_name,time_stamp))
        torch.save(state, save_path)
        print("Saving checkpoint '{}_{}_best_model.pkl' (epoch {})".format(exp_name,time_stamp, epoch))
                        
om = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
rgb = decode_segmap(om[0])
plt.imshow(rgb)
