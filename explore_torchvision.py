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
import torch.optim as optim
from torch import nn

from models.encoder_decoder import get_encoder_decoder
from utils.metrics import runningScore, averageMeter
from utils.loss import cross_entropy2d
from utils.im_utils import decode_segmap, convert_targets, cat_labels, imshow
from dataloader import get_dataloaders 

def train(cfg):
    
    gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder_name = cfg["model"]["encoder"]
    decoder_name = cfg["model"]["decoder"]
    
    num_classes = cfg['tasks']['semantic']['classes']
    
    train_params = cfg['params']['train']        
    epochs = train_params["epochs"]
    resume_training = train_params["resume"]
    patience = train_params["patience"]
    early_stop = train_params["early_stop"]
    
    base_dir =  os.path.join(os.path.expanduser('~'),'results')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    exp_name  =  (encoder_name + '-' + str(decoder_name) +
                  '-' + str(512) + '-' + '_'.join(cfg['tasks'].keys()))
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  

    dataloaders = get_dataloaders(cfg)
    
    model = get_encoder_decoder(cfg)
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    
    

    optimizer = optim.Adam(model.parameters(), lr=0.0001)    
    
    
    
    best_iou = -100.0
    start_iter = 0
    plateau_count = 0
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
    
    train_loss_meter = averageMeter()
    val_loss_meter   = averageMeter()
    time_meter       = averageMeter()
    running_metrics_val = runningScore(num_classes)
        
    for epoch in range(epochs):
        print('********************** '+str(epoch+1)+' **********************')
        for i, data in tqdm(enumerate(dataloaders['train'])):
            t = time.time()
            
            inputs,targets = data
            
            rgb_targets = convert_targets(targets)

            optimizer.zero_grad()
            
            outputs = model(inputs.to(device))
            rgb_loss = cross_entropy2d(outputs, rgb_targets.long().to(device),weight=class_weights)
        
            loss = rgb_loss
        
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
            for i,data in tqdm(enumerate(dataloaders['val'])):
                #if i%10 == 9:
                #    break
                images, targets = data
                rgb_targets = convert_targets(targets)
                #dep_targets = targets[1]
            
                outputs  = model(images.to(device))
                rgb_val_loss = cross_entropy2d(outputs, rgb_targets.long().to(device),weight=class_weights)
                #dep_val_loss = huber_loss(outputs[1], dep_targets.float().to(device))
        
                val_loss = rgb_val_loss# + dep_val_loss
                
                pred = outputs.data.max(1)[1].cpu().numpy()
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
                #om = torch.argmax(outputs[0].squeeze(), dim=1).detach().cpu().numpy()
                #rgb = decode_segmap(om[0],nc=n_classes)
                #mask = (gt < n_classes)[0]
                #mask = np.repeat(np.expand_dims(mask,axis=-1),3,axis=-1)
                #plt.imshow(rgb*mask)
                #plt.show()
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