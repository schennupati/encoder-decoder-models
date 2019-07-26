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

from data_loaders import Cityscapes
from torch import nn
from torchvision import transforms

from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, transform_targets, convert_targets, cat_labels
#from utils.im_utils import get_class_weights

gpus = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder_name = 'resnet101'
decoder_name = 'fpn'
tasks = {'seg':19,'dep':1}
im_size = 512
batch_size = 4 
best_iou = -100.0
resume_training = True

base_dir =  os.path.join(os.path.expanduser('~'),'results')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

exp_name  =  (encoder_name + '-' + str(decoder_name) +
             '-' + str(im_size) + '-' + '_'.join(tasks.keys()))
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

model = get_encoder_decoder(encoder_name, decoder_name, tasks=tasks)
model = model.to(device)

print(model)

if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

running_metrics_val = runningScore(tasks['seg'])

train_loss_meter = averageMeter()
val_loss_meter   = averageMeter()
time_meter       = averageMeter()
    
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#class_weights = get_class_weights(train_loader,n_classes)
class_weights = [3.045383480249677, 12.862127312658735, 4.509888876996228, 
                 38.15694593009221, 35.25278401818165, 31.48260832348194, 
                 45.79224481584843, 39.69406346608758, 6.0639281852733715, 
                 32.16484408952653, 17.10923371690307, 31.5633201415795, 
                 47.33397232867321, 11.610673599796504, 44.60042610251128, 
                 45.23705196392834, 45.28288297518183, 48.14776939659858, 
                 41.924631833506794] #CityScapes 19 Classes weights

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
        loss = cross_entropy2d(outputs[0], targets.long().to(device),weight=class_weights)
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
            val_loss = cross_entropy2d(outputs[0], targets.long().to(device),weight=class_weights)
            pred = outputs[0].data.max(1)[1].cpu().numpy()
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
                        
om = torch.argmax(outputs[0].squeeze(), dim=1).detach().cpu().numpy()
rgb = decode_segmap(om[0])
plt.imshow(rgb)
om = outputs[1].squeeze().detach().cpu().numpy()
plt.imshow(om[0])
