#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:16:32 2019

@author: sumche
"""
import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.data_utils import convert_targets, convert_outputs, post_process_outputs
from utils.metrics import metrics
from utils.loss_utils import compute_loss, loss_meters
from utils.im_utils import cat_labels,decode_segmap
from utils.checkpoint_loader import get_checkpoint

import matplotlib.pyplot as plt

def get_device(cfg):
    #TODO: Fetch device using cfg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_exp_name(cfg):
    encoder_name = cfg["model"]["encoder"]
    decoder_name = cfg["model"]["decoder"]
    imsize  = cfg['data']['im_size']
    base_dir = cfg["model"]["pretrained_path"]
    ###### Define Experiment save path ######
    if base_dir is None:
        base_dir =  os.path.join(os.path.expanduser('~'),'results')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    return encoder_name + '-' + decoder_name +'-' + str(imsize) + '-' + '_'.join(cfg['tasks'].keys())
    
def get_save_path(cfg):    
    base_dir = cfg["model"]["pretrained_path"]
    exp_name = get_exp_name(cfg) 
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    path = os.path.join(base_dir,"{}_{}_best_model.pkl".format(exp_name,time_stamp))
     
    return path,exp_name,time_stamp

def get_writer(cfg):
    base_dir = cfg["model"]["pretrained_path"]
    _,exp_name,time_stamp = get_save_path(cfg)
    log_dir = os.path.join(base_dir,'logs',"{}_{}".format(exp_name,time_stamp))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return SummaryWriter(log_dir=log_dir)

def get_config_params(cfg):
        
    params = cfg['params']
    base_dir = cfg["model"]["pretrained_path"]
    exp_name = get_exp_name(cfg)
    resume_training = params["resume"]        
    epochs = params["epochs"]
    patience = params["patience"]
    early_stop = params["early_stop"]
    print_interval = params['print_interval']
    best_loss = 1e10
    start_iter = 0
    plateau_count = 0
    state = None
    
    return params,epochs,patience,early_stop,base_dir,exp_name,resume_training,\
           print_interval,best_loss,start_iter,plateau_count,state

def if_checkpoint_exists(exp_name,base_dir):
    if get_checkpoint(exp_name,base_dir) is not None:
        return True
    else:
        return False

def load_checkpoint(model,optimizer,exp_name,base_dir):
    ###### load pre trained models ######
    checkpoint_name,checkpoint = get_checkpoint(exp_name,base_dir)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_iter = checkpoint["epoch"]
    best_loss = checkpoint['best_loss']
    print("Loaded checkpoint '{}' from epoch {} with loss {}".format(checkpoint_name, start_iter, best_loss))
    
    return model,optimizer,start_iter,best_loss
        
def init_optimizer(model,params):
    optimizer_cls = get_optimizer(params['train'])
    optimizer_params = {k: v for k, v in params['train']["optimizer"].items() if k != "name"}      
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    
    return optimizer
    
def get_model(cfg,device):
    gpus = list(range(torch.cuda.device_count()))
    model = get_encoder_decoder(cfg)
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    return model

def get_losses_and_metrics(cfg):
    train_loss_meters = loss_meters(cfg['tasks'])
    val_loss_meters   = loss_meters(cfg['tasks'])
    val_metrics       = metrics(cfg['tasks']) 
    
    return train_loss_meters, val_loss_meters, val_metrics

def train_step(model,data,optimizer,cfg,device,weights,running_loss,
               train_loss_meters,print_interval,n_steps,epoch,step,writer):
    
    optimizer.zero_grad()
    inputs,targets = data 
    outputs = model(inputs.to(device))
    outputs = convert_outputs(outputs,cfg['tasks'])
    targets = convert_targets(targets,cfg['tasks'])
    losses,loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
    running_loss.update(loss)
    loss.backward()
    optimizer.step()
    train_loss_meters.update(losses)
    
    if step % print_interval == print_interval - 1 or step == n_steps-1:
        print("\nepoch: {} batch: {} loss: {}".format(epoch + 1, step + 1, running_loss.avg))
        writer.add_scalar('Loss/train', running_loss.avg, epoch*n_steps + step)
        for k, v in train_loss_meters.meters.items():
            print("{} loss: {}".format(k, v.avg))
            writer.add_scalar('Loss/train_{}'.format(k), v.avg, epoch*n_steps + step)
            
    running_loss.reset()
    train_loss_meters.reset()
    
    return None
    
def validation_step(model,dataloaders,cfg,device,weights,running_val_loss,
                    val_loss_meters,val_metrics,epoch,writer):
    print('\n********************* validation ********************') 
    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloaders['val'])):
            inputs,targets = data 
            outputs = model(inputs.to(device))
            outputs     = convert_outputs(outputs,cfg['tasks'])
            predictions = post_process_outputs(outputs,cfg['tasks'])
            targets     = convert_targets(targets,cfg['tasks'])
            val_losses,val_loss = compute_loss(outputs,targets,cfg['tasks'],device,weights)
            val_loss_meters.update(val_losses)
            running_val_loss.update(val_loss)
        
        print("\nepoch: {} validation_loss: {}".format(epoch + 1, running_val_loss.avg))
        writer.add_scalar('Loss/Val', running_val_loss.avg, epoch)
        idx = np.random.random_integers(i)
        for k, v in val_loss_meters.meters.items():
            print("{} loss: {}".format(k, v.avg))
            writer.add_scalar('Loss/Validation_{}'.format(k), v.avg, epoch)
            add_images_to_writer(predictions,writer,k,idx,epoch)

        current_loss = running_val_loss.avg
        running_val_loss.reset()
        val_loss_meters.reset()
        val_metrics.update(targets, predictions)
    
    return val_metrics, current_loss
        
def print_metrics(val_metrics):
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

def save_model(model,optimizer,cfg,current_loss,best_loss,plateau_count,start_iter,epoch,state):
    if current_loss <= best_loss:
        best_loss = current_loss
        state = {"epoch": start_iter + epoch + 1,"model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),"best_loss": best_loss}
        save_path,exp_name,time_stamp = get_save_path(cfg)
        torch.save(state, save_path)
        print("\nSaving checkpoint '{}_{}_best_model.pkl' (epoch {})".format(exp_name,time_stamp, epoch+1))
        plateau_count = 0
    else:
        plateau_count +=1
        
    return state,best_loss,plateau_count

def stop_training(patience,plateau_count,early_stop,epoch,state):
    if plateau_count == patience and early_stop:
        print('Early Stopping after {} epochs: Patience of {} epochs reached.'.format(epoch+1,plateau_count))
        print('Best Checkpoint:')
        for k, v in state.items():
            print("{} ({})".format(k, v))
            
def add_images_to_writer(predictions,writer,task,idx,epoch):
    if task == 'semantic':
        img = decode_segmap(predictions[task][idx,:,:,:])
        writer.add_figure('Images/validation_{}'.format(task),
                          plt.imshow(img),epoch)
    elif task == 'instance_cluster':
        x_img = predictions[task][idx,1,:,:]
        y_img = predictions[task][idx,0,:,:]
        writer.add_figure('Images/validation_{}_dx'.format(task),
                          plt.imshow(x_img),epoch)
        writer.add_figure('Images/validation_{}_dy'.format(task),
                          plt.imshow(y_img),epoch)
        
    
    

    
    