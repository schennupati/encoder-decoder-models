#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""

import time
import os
import matplotlib.pyplot as plt
import datetime
import glob
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from datasets.cityscapes import Cityscapes


from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, convert_targets, cat_labels, imshow, cityscapes_class_weights

gpus = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder_name = 'resnet101'
decoder_name = 'fpn'
tasks = {'seg':20}#,'dep':1}
im_size = 512
full_res = 1024
batch_size = 4 
best_iou = -100.0
resume_training = False
epochs = 100
patience = 5
early_stop = True
ignore_last = True
n_classes = tasks['seg']

base_dir =  os.path.join(os.path.expanduser('~'),'results')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

exp_name  =  (encoder_name + '-' + str(decoder_name) +
             '-' + str(im_size) + '-' + '_'.join(tasks.keys()))
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

train_transform = transforms.Compose([transforms.RandomCrop((im_size,2*im_size)),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_target_transform = transforms.Compose([transforms.RandomCrop((im_size,2*im_size)),transforms.ToTensor()])

val_transform = transforms.Compose([transforms.Resize(full_res),transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_target_transform = transforms.Compose([transforms.Resize(full_res),transforms.ToTensor()])

train_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                           target_type='semantic',transform=train_transform,
                           target_transform=train_target_transform)

val_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='val', mode='fine',
                         target_type='semantic',transform=val_transform,
                         target_transform=val_target_transform)

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

targets = convert_targets(targets)
rgb = decode_segmap(targets[0].numpy(),nc=n_classes)
imshow(data[0])
plt.imshow(rgb)
plt.show()

model = get_encoder_decoder(encoder_name, decoder_name, tasks=tasks)
model = model.to(device)

#print(model)

if len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus, dim=0)

train_loss_meter = averageMeter()
val_loss_meter   = averageMeter()
time_meter       = averageMeter()
running_metrics_val = runningScore(n_classes,ignore_last=ignore_last)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

class_weights = cityscapes_class_weights(n_classes)

class_weights = torch.FloatTensor(class_weights).cuda()

list_of_models = glob.glob(os.path.join(base_dir,'*.pkl'))

if any(exp_name in model for model in list_of_models) and resume_training:
    latest_model = max(list_of_models, key=os.path.getctime)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_iter = checkpoint["epoch"]
    best_iou = checkpoint['best_iou']
    print("Loaded checkpoint '{}' from epoch {} with mIoU {}".format(latest_model, start_iter, best_iou))
    
else:
    print("Begining Training from Scratch")

plateau_count = 0

for epoch in range(epochs):
    print('********************** '+str(epoch+1)+' **********************')
    for i, data in tqdm(enumerate(train_loader, 0)):
        t = time.time()
        
        inputs,targets = data
        
        targets = convert_targets(targets)
                
        optimizer.zero_grad()
        
        outputs = model(inputs.to(device))
        loss = cross_entropy2d(outputs[0], targets.long().to(device),weight=class_weights)
        loss.backward()
        optimizer.step()
        
        time_meter.update(time.time() - t)        
        train_loss_meter.update(loss.item())
        
        if i % 10 == 9:        
            print('epoch: %d batch: %d time_per_batch: %.3f  loss: %.3f' %
                      (epoch + 1, i + 1, time_meter.avg , train_loss_meter.avg))
            running_loss = 0.0
            train_loss_meter.reset()
            time_meter.reset()
    with torch.no_grad():
        for i,data in tqdm(enumerate(val_loader)):
            images, targets = data
            targets = convert_targets(targets)
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
        state = {"epoch": start_iter + epoch + 1,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "best_iou": best_iou}
        save_path = os.path.join(base_dir,"{}_{}_best_model.pkl".format(exp_name,time_stamp))
        torch.save(state, save_path)
        print("Saving checkpoint '{}_{}_best_model.pkl' (epoch {})".format(exp_name,time_stamp, epoch+1))
        plateau_count = 0
        imshow(images[0])
        om = torch.argmax(outputs[0].squeeze(), dim=1).detach().cpu().numpy()
        rgb = decode_segmap(om[0],nc=n_classes)
        plt.imshow(rgb)
        plt.show()
    else:
        plateau_count +=1
    
    if plateau_count == patience and early_stop:
        print('Early Stopping after {} epochs: Patience of {} epochs reached.'.format(epoch+1,plateau_count))
        break 

