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
import torchprof

from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.data_utils import convert_targets, convert_outputs, post_process_outputs, get_weights
from utils.metrics import metrics
from utils.loss_utils import compute_loss, loss_meters, averageMeter
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
    imsize  = cfg['data']['im_size']
    params = cfg['params']        
    epochs = params["epochs"]
    resume_training = params["resume"]
    patience = params["patience"]
    early_stop = params["early_stop"]
    print_interval = params['print_interval']
    best_loss = 1e10
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
    data = iter(dataloaders['train']).next()[0].to(device)
    #print(model)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    with torchprof.Profile(model, use_cuda=True) as prof:
        model(data)
    print(prof.display(show_events=False))
        
    optimizer_cls = get_optimizer(params['train'])
    optimizer_params = {k: v for k, v in params['train']["optimizer"].items() if k != "name"}      
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    
    train_loss_meters = loss_meters(cfg['tasks'])
    val_loss_meters   = loss_meters(cfg['tasks'])
    val_metrics = metrics(cfg['tasks'])    
    
    ###### load pre trained models ######
    if resume_training and get_checkpoint(exp_name,base_dir) is not None:
        checkpoint_name,checkpoint = get_checkpoint(exp_name,base_dir)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = checkpoint["epoch"]
        best_loss = checkpoint['best_loss']
        print("Loaded checkpoint '{}' from epoch {} with loss {}".format(checkpoint_name, start_iter, best_loss))
        
    else:
        print("Begining Training from Scratch")    
            
    for epoch in range(epochs):
        print('\n********************** Epoch {} **********************'.format(epoch+1))
        print('********************** Training *********************')
        running_loss = averageMeter()
        for i, data in tqdm(enumerate(dataloaders['train'])):
            optimizer.zero_grad()
            
            inputs,targets = data 
            outputs = model(inputs.to(device))
            params = None
            
            if isinstance(outputs,tuple):
                params  = outputs[0]
                outputs = outputs[1]
                
            
            outputs = convert_outputs(outputs,cfg['tasks'])
            targets = convert_targets(targets,cfg['tasks'])
            
            losses,loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
            running_loss.update(loss)
            #make_dot(loss).view()
            loss.backward()
            optimizer.step()
            train_loss_meters.update(losses)
            if i % print_interval == print_interval - 1 or i == len(dataloaders['train'])-1:
                print("\nepoch: {} batch: {} loss: {}".format(epoch + 1, i + 1, running_loss.avg))
                for k, v in train_loss_meters.meters.items():
                    print("{} loss: {}".format(k, v.avg))
                if params is not None:
                    print('**************** Cross-Stitch Parameters ***************')
                    for param in params:
                        param = param.data.cpu().numpy()
                        print(param)#/param.sum(axis=0,keepdims=1))
                #break
            running_loss.reset()
            train_loss_meters.reset()
        with torch.no_grad():
            running_val_loss = averageMeter()
            print('\n********************* validation ********************')
            for i,data in tqdm(enumerate(dataloaders['val'])):
                #if i%10 == 9:
                #    break
                inputs,targets = data 
                outputs = model(inputs.to(device))
                
                if isinstance(outputs,tuple):
                    params  = outputs[0]
                    outputs = outputs[1]

                outputs     = convert_outputs(outputs,cfg['tasks'])
                predictions = post_process_outputs(outputs,cfg['tasks'])
                targets     = convert_targets(targets,cfg['tasks'])
                
                val_losses,val_loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
                val_loss_meters.update(val_losses)
                running_val_loss.update(val_loss)
            print("\nepoch: {} validation_loss: {}".format(epoch + 1, running_val_loss.avg))
            for k, v in val_loss_meters.meters.items():
                print("{} loss: {}".format(k, v.avg))
            current_loss = running_val_loss.avg
            running_val_loss.reset()
            val_loss_meters.reset()
            val_metrics.update(targets, predictions)
            
            print('\n********************** Metrics *********************')
            for task in val_metrics.metrics.keys():
                if val_metrics.metrics[task] is not None:
                    score, class_iou = val_metrics.metrics[task].get_scores()
                    print('********************** {} *********************'.format(task))
                    if isinstance(score,dict):
                        [print(k,'\t :',v) for k,v in score.items() ]
                    if task =='semantic':
                        [print(cat_labels[k].name,'\t :',v) for k,v in class_iou.items() ]
                        
            val_metrics.reset()
            if current_loss <= best_loss:
                best_loss = current_loss
                state = {"epoch": start_iter + epoch + 1,
                         "model_state": model.state_dict(),
                         "optimizer_state": optimizer.state_dict(),
                         "best_loss": best_loss}
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