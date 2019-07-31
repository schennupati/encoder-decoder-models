#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import argparse
import os
import datetime
import yaml
from tqdm import tqdm

import torch
from torch import nn

from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.preprocess import convert_targets, convert_outputs, get_weights
from utils.metrics import runningScore
from utils.loss_utils import compute_loss, loss_meters
from utils.im_utils import cat_labels, imshow
from utils.dataloader import get_dataloaders 
from utils.checkpoint_loader import get_checkpoint

def train(cfg):
    
    gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ###### Define Configuration parameters #####

    encoder_name = cfg["model"]["encoder"]
    decoder_name = cfg["model"]["decoder"]
    base_dir = cfg["model"]["pretrained_path"]
    num_classes = cfg['tasks']['semantic']['out_channels']
    imsize  = cfg['data']['im_size']
    params = cfg['params']        
    epochs = params["epochs"]
    resume_training = params["resume"]
    patience = params["patience"]
    early_stop = params["early_stop"]
    print_interval = params['print_interval']
    best_iou = -100.0
    start_iter = 0
    plateau_count = 0
    
    weights = get_weights(cfg)
    
    ###### Define Experiment save path ######
    if base_dir is None:
        base_dir =  os.path.join(os.path.expanduser('~'),'results')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    exp_name  =  (encoder_name + '-' + decoder_name +
                  '-' + str(imsize) + '-' + '_'.join(cfg['tasks'].keys()))
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')  
    
    ###### Define dataloaders, model, optimizers and metrics######
    dataloaders = get_dataloaders(cfg)
    model = get_encoder_decoder(cfg)
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
        
    optimizer_cls = get_optimizer(params['train'])
    optimizer_params = {k: v for k, v in params['train']["optimizer"].items() if k != "name"}      
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    
    train_loss_meters = loss_meters(cfg['tasks'])
    val_loss_meters   = loss_meters(cfg['tasks'])
    running_metrics_val = runningScore(num_classes)    
    
    ###### load pre trained models ######
    if resume_training and get_checkpoint(exp_name,base_dir) is not None:
        checkpoint_name,checkpoint = get_checkpoint(exp_name,base_dir)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = checkpoint["epoch"]
        best_iou = checkpoint['best_iou']
        print("Loaded checkpoint '{}' from epoch {} with mIoU {}".format(checkpoint_name, start_iter, best_iou))
        
    else:
        print("Begining Training from Scratch")    
            
    for epoch in range(epochs):
        print('\n********************** Epoch {} **********************'.format(epoch+1))
        print('********************** Training *********************')
        for i, data in tqdm(enumerate(dataloaders['train'])):
            optimizer.zero_grad()
            
            inputs,targets = data 
            outputs = model(inputs.to(device))
            
            outputs = convert_outputs(outputs,cfg['tasks'])
            targets = convert_targets(targets,cfg['tasks'])
            
            losses,loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
            loss.backward()
            optimizer.step()
            train_loss_meters.update(losses)
            if i % print_interval == print_interval - 1 or i == len(dataloaders['train'])-1:
                print("\nepoch: {} batch: {}".format(epoch + 1, i + 1))
                for k, v in train_loss_meters.meters.items():
                    print("{} loss: {}".format(k, v.avg))
                train_loss_meters.reset()
                #break
        with torch.no_grad():
            print('\n\n********************* validation ********************')
            for i,data in tqdm(enumerate(dataloaders['val'])):
                #if i%10 == 9:
                #    break
                inputs,targets = data 
                outputs = model(inputs.to(device))
                
                outputs = convert_outputs(outputs,cfg['tasks'])
                targets = convert_targets(targets,cfg['tasks'])
                
                val_losses,val_loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
                val_loss_meters.update(val_losses)
                if i % print_interval == print_interval - 1 or i == len(dataloaders['val'])-1:
                    print("\nbatch: {}".format( i + 1))
                    for k, v in val_loss_meters.meters.items():
                        print("{} loss: {}".format(k, v.avg))
                    val_loss_meters.reset()
                
                pred = outputs['semantic'].data.max(1)[1].cpu().numpy()
                gt = targets['semantic'].data.cpu().numpy()
                running_metrics_val.update(gt, pred)
            score, class_iou = running_metrics_val.get_scores()
            
            for k,v in score.items():
                print(k,v)
            for k,v in class_iou.items():
                print(cat_labels[k].name,': ',v)
            
            running_metrics_val.reset()
            val_loss_meters.reset()
            if score["Mean IoU : \t"] >= best_iou:
                best_iou = score["Mean IoU : \t"]
                state = {"epoch": start_iter + epoch + 1,
                         "model_state": model.state_dict(),
                         "optimizer_state": optimizer.state_dict(),
                         "best_iou": best_iou}
                save_path = os.path.join(base_dir,"{}_{}_best_model.pkl".format(exp_name,time_stamp))
                torch.save(state, save_path)
                print("\nSaving checkpoint '{}_{}_best_model.pkl' (epoch {})".format(exp_name,time_stamp, epoch+1))
                plateau_count = 0
            else:
                plateau_count +=1
    
    if plateau_count == patience and early_stop:
        print('Early Stopping after {} epochs: Patience of {} epochs reached.'.format(epoch+1,plateau_count))
        print('Best Checkpoint:')
        for k, v in state.items():
            print("{} ({})".format(k, v))

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
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
        
    train(cfg)