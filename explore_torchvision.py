#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""

import argparse
import time
import os
import matplotlib.pyplot as plt
import datetime
import glob
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from datasets.cityscapes import Cityscapes

from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, convert_targets, cat_labels, imshow, RandomScale 

def train(cfg):
    
    gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder_name = cfg["model"]["encoder"]
    decoder_name = cfg["model"]["decoder"]
    im_size  = cfg["model"]["input_im_size"]
    full_res = cfg["model"]["val_im_size"]
    
    tasks = cfg["tasks"]
    
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    resume_training = cfg["training"]["resume"]
    patience = cfg["training"]["patience"]
    early_stop = cfg["training"]["early_stop"]
    
    best_iou = -100.0
    start_iter = 0
    plateau_count = 0
    
    n_classes = tasks['seg']
    
    base_dir =  os.path.join(os.path.expanduser('~'),'results')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    exp_name  =  (encoder_name + '-' + str(decoder_name) +
                  '-' + str(im_size) + '-' + '_'.join(tasks.keys()))
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    train_transform = transforms.Compose([RandomScale(scale=(0.5,2.0)),transforms.RandomCrop((im_size,2*im_size)),
                                      transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_target_transform = transforms.Compose([RandomScale(scale=(0.5,2.0)),transforms.RandomCrop((im_size,2*im_size)),
                                             transforms.ToTensor()])
    
    #transforms.Resize(im_size, interpolation=2)
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #transforms.RandomResizedCrop((im_size,2*im_size),scale=(0.5,2.0))
    #transforms.RandomCrop((im_size,2*im_size))

    val_transform = transforms.Compose([transforms.Resize(full_res),transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_target_transform = transforms.Compose([transforms.Resize(full_res),transforms.ToTensor()])

    train_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='train', mode='fine',
                           target_type=['semantic'],transform=train_transform,
                           target_transform=train_target_transform)

    val_dataset = Cityscapes('/home/sumche/datasets/Cityscapes', split='val', mode='fine',
                         target_type=['semantic'],transform=val_transform,
                         target_transform=val_target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)
    
    for _ in range(1):
        dataiter = iter(train_loader)
        data,targets = dataiter.next()
        #dep = convert_targets_instance(targets[1])
        #dep = dep[0]
        #print(torch.unique(dep))
        targets = convert_targets(targets)
        rgb = decode_segmap(targets[0].numpy(),nc=n_classes)
        imshow(data[0])
        plt.imshow(rgb)
        plt.show()
        #plt.imshow(dep)
        #plt.show()    
    
    model = get_encoder_decoder(encoder_name, decoder_name, tasks=tasks)
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    
    train_loss_meter = averageMeter()
    val_loss_meter   = averageMeter()
    time_meter       = averageMeter()
    running_metrics_val = runningScore(n_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)    
    
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
        
    class_weights = [3.045383480249677, 12.862127312658735, 4.509888876996228, 
                     38.15694593009221, 35.25278401818165, 31.48260832348194, 
                     45.79224481584843, 39.69406346608758, 6.0639281852733715, 
                     32.16484408952653, 17.10923371690307, 31.5633201415795, 
                     47.33397232867321, 11.610673599796504, 44.60042610251128, 
                     45.23705196392834, 45.28288297518183, 48.14776939659858, 
                     41.924631833506794]

    class_weights = torch.FloatTensor(class_weights).cuda()
        
    for epoch in range(epochs):
        print('********************** '+str(epoch+1)+' **********************')
        for i, data in tqdm(enumerate(train_loader, 0)):
            t = time.time()
            
            inputs,targets = data
            
            rgb_targets = convert_targets(targets)
            #dep_targets = targets[1]

            optimizer.zero_grad()
            
            outputs = model(inputs.to(device))
            rgb_loss = cross_entropy2d(outputs[0], rgb_targets.long().to(device),weight=class_weights)
            #dep_loss = huber_loss(outputs[1], dep_targets.float().to(device))
        
            loss = rgb_loss# + dep_loss
        
            loss.backward()
            optimizer.step()
        
            time_meter.update(time.time() - t)        
            train_loss_meter.update(loss.item())
        
            if i % 10 == 9:        
                print('epoch: %d batch: %d time_per_batch: %.3f  loss: %.3f' %
                          (epoch + 1, i + 1, time_meter.avg , train_loss_meter.avg))

                train_loss_meter.reset()
                time_meter.reset()
                #break
        
        with torch.no_grad():
            for i,data in tqdm(enumerate(val_loader)):
                #if i%10 == 9:
                #    break
                images, targets = data
                rgb_targets = convert_targets(targets)
                #dep_targets = targets[1]
            
                outputs  = model(images.to(device))
                rgb_val_loss = cross_entropy2d(outputs[0], rgb_targets.long().to(device),weight=class_weights)
                #dep_val_loss = huber_loss(outputs[1], dep_targets.float().to(device))
        
                val_loss = rgb_val_loss# + dep_val_loss
                
                pred = outputs[0].data.max(1)[1].cpu().numpy()
                gt = rgb_targets.data.cpu().numpy()
            
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
                mask = (gt < n_classes)[0]
                mask = np.repeat(np.expand_dims(mask,axis=-1),3,axis=-1)
                plt.imshow(rgb*mask)
                plt.show()
            else:
                plateau_count +=1
    
    if plateau_count == patience and early_stop:
        print('Early Stopping after {} epochs: Patience of {} epochs reached.'.format(epoch+1,plateau_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
            "--config",
            nargs="?",
            type=str,
            default="configs/base.yml",
            help="Configuration to use")
    args = parser.parse_args()
    
    with open(args.config) as fp:
        cfg = yaml.load(fp)
        
    train(cfg)