#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:16:32 2019

@author: sumche
"""
import os
import glob
import datetime
import numpy as np
from tqdm import tqdm
import pdb
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import io
import tensorflow as tf
from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.data_utils import (convert_targets, convert_outputs, 
                              post_process_outputs, getPanopticFromContour)
from utils.metrics import metrics
from utils.loss_utils import compute_loss, loss_meters, MultiTaskLoss
from utils.im_utils import cat_labels, prob_labels, inst_labels, decode_segmap
from PIL import Image
import matplotlib.pyplot as plt
from instance_to_clusters import get_clusters, imshow_components, to_rgb
import cv2

def get_device(cfg):
    device_str = 'cuda:{}'.format(cfg['params']['gpu_id']) if not cfg['params']['multigpu'] else "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    return device

def get_exp_dir(cfg):
    encoder_name = cfg["model"]["encoder"]
    decoder_name = cfg["model"]["decoder"]
    imsize  = cfg['data']['im_size']
    base_dir = cfg["model"]["pretrained_path"]
    ###### Define Experiment save path ######
    if base_dir is None:
        base_dir =  os.path.join(os.path.expanduser('~'),'results')
    exp_name = (encoder_name + '-' + decoder_name +'-' + str(imsize) ) 
    task_name = '_'.join(cfg['tasks'].keys())
    exp_dir_path = os.path.join(base_dir,task_name,exp_name)
    
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)
        
    return exp_dir_path,task_name,exp_name
    
def get_save_path(cfg,best_loss=None):    
    exp_dir_path,task_name,exp_name = get_exp_dir(cfg) 
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    path = os.path.join(exp_dir_path,"{}_best-loss_{}.pkl".format(time_stamp,best_loss))
     
    return path,exp_name,task_name,time_stamp

def get_writer(cfg):
    base_dir = cfg["model"]["pretrained_path"]
    savelogs = cfg["params"]["savelogs"]
    _,exp_name,task_name,time_stamp = get_save_path(cfg)
    log_dir = os.path.join(base_dir,'logs',task_name,exp_name,"{}".format(time_stamp))
    if not os.path.exists(log_dir) and savelogs:
        os.makedirs(log_dir)
        return SummaryWriter(log_dir=log_dir)
    else:
        return None

def get_config_params(cfg):
        
    params = cfg['params']
    #base_dir = cfg["model"]["pretrained_path"]
    exp_dir,_,_ = get_exp_dir(cfg)
    resume_training = params["resume"]        
    epochs = params["epochs"]
    patience = params["patience"]
    early_stop = params["early_stop"]
    print_interval = params['print_interval']
    best_loss = 1e6
    start_iter = 0
    plateau_count = 0
    state = None
    
    return params,epochs,patience,early_stop,exp_dir,resume_training,\
           print_interval,best_loss,start_iter,plateau_count,state

def get_best_model(list_of_models):
    best_loss  = 1e6
    best_model = None
    for model in list_of_models:
        checkpoint_name = model.split('/')[-1].split('.pkl')[0]
        loss = float(checkpoint_name.split('_')[-1])
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model
    
def get_checkpoint_list(exp_dir):
    return glob.glob(os.path.join(exp_dir,'*.pkl'))

def get_checkpoint(exp_dir):
    list_of_models = get_checkpoint_list(exp_dir)
    checkpoint_name = get_best_model(list_of_models)
    return checkpoint_name, torch.load(checkpoint_name)

def if_checkpoint_exists(exp_dir):
    list_of_models = get_checkpoint_list(exp_dir)
    if len(list_of_models) != 0:
        return True
    else:
        return False

def load_checkpoint(model,optimizer,criterion,base_dir):
    ###### load pre trained models ######
    checkpoint_name,checkpoint = get_checkpoint(base_dir)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    criterion.load_state_dict(checkpoint["criterion_state"])
    start_iter = checkpoint["epoch"]
    best_loss = checkpoint['best_loss']
    print("Loaded checkpoint '{}' from epoch {} with loss {}".format(checkpoint_name, start_iter, best_loss))
    
    return model,optimizer,criterion,start_iter,best_loss

def get_criterion(cfg, weights):
    loss_fn = cfg['model']['loss_fn']
    criterion, params = None, None
    criterion = MultiTaskLoss(cfg['model']['outputs'], weights, loss_fn)
    params = list(criterion.parameters())
    return criterion, params
        
def init_optimizer(model, params, criterion, loss_params):
    optimizer_cls = get_optimizer(params['train'])
    optimizer_params = {k: v for k, v in params['train']["optimizer"].items() if k != "name"}
    if loss_params is not None:      
        model_params = list(model.parameters()) + loss_params
    else:
        model_params = model.parameters()
    optimizer = optimizer_cls(model_params, **optimizer_params)
    
    return optimizer

def get_model(cfg,device):
    gpus = list(range(torch.cuda.device_count()))
    model = get_encoder_decoder(cfg)
    model = model.to(device)
    multigpu = cfg['params']['multigpu']
    if len(gpus) > 1 and multigpu:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    return model

def get_losses_and_metrics(cfg):
    train_loss_meters = loss_meters(cfg['model']['outputs'])
    val_loss_meters   = loss_meters(cfg['model']['outputs'])
    val_metrics       = metrics(cfg['model']['outputs']) 
    
    return train_loss_meters, val_loss_meters, val_metrics

def train_step(model,data,optimizer,cfg,device,weights,running_loss,
               train_loss_meters, criterion, print_interval, 
               n_steps,epoch,step,writer):
    
    optimizer.zero_grad()
    inputs,targets = data 
    outputs = model(inputs.to(device))
    targets = convert_targets(targets,cfg['tasks'],device)
    losses, loss = criterion(outputs,targets)
    #pdb.set_trace()
    running_loss.update(loss)
    loss.backward()
    optimizer.step()
    train_loss_meters.update(losses)
    
    if (step % print_interval == 0 or step == n_steps-1):
        loss_weights = [p.data for p in criterion.parameters()]
        print('\nLoss weights {}'.format(loss_weights))
        if writer:
            writer.add_scalar('Loss/train', running_loss.avg, epoch*n_steps + step)
        loss_str = "epoch: {} batch: {} loss: {}".format(epoch + 1, step , running_loss.avg)
        predictions = post_process_outputs(outputs,cfg['model']['outputs'], targets)
        
        for k, v in train_loss_meters.meters.items():
            loss_str += " {}_loss: {}".format(k, v.avg)
            if writer:
                writer.add_scalar('Loss/train_{}'.format(k), v.avg, epoch*n_steps + step)
                #if step == 0:
                add_images_to_writer(inputs,outputs,predictions,targets,writer,k,epoch,train=True)
        print(loss_str)
    running_loss.reset()
    train_loss_meters.reset()
    
    return None
    
def validation_step(model,dataloaders,cfg,device,weights,running_val_loss,
                    val_loss_meters, criterion, val_metrics,epoch,writer):
    print('\n********************* validation ********************') 
    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloaders['val'])):
            inputs,targets = data 
            outputs = model(inputs.to(device))
            targets = convert_targets(targets,cfg['tasks'],device)
            #pdb.set_trace()
            val_losses, val_loss = criterion(outputs,targets)
            val_loss_meters.update(val_losses)
            running_val_loss.update(val_loss)
            predictions = post_process_outputs(outputs,cfg['model']['outputs'], targets)
            val_metrics.update(targets, predictions)
            #print_metrics(val_metrics)
        print("\nepoch: {} validation_loss: {}".format(epoch + 1, running_val_loss.avg))
        if writer:
            writer.add_scalar('Loss/Val', running_val_loss.avg, epoch)
        for k, v in val_loss_meters.meters.items():
            print("{} loss: {}".format(k, v.avg))
            if writer:
                writer.add_scalar('Loss/Validation_{}'.format(k), v.avg, epoch)
                add_images_to_writer(inputs,outputs,predictions,targets,writer,k,epoch)
            
        current_loss = running_val_loss.avg
        running_val_loss.reset()
        val_loss_meters.reset()
        
    
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
                elif task =='instance_probs':
                    [print(prob_labels[k].name,'\t :',v) for k,v in class_iou.items() ]
                elif task =='instance_contour':
                    [print(inst_labels[k].name,'\t :',v) for k,v in class_iou.items() ]


def save_model(model,optimizer,criterion,cfg,current_loss,best_loss,plateau_count,start_iter,epoch,state):
    if current_loss <= best_loss:
        best_loss = current_loss
        state = {"epoch": start_iter + epoch + 1,"model_state": model.state_dict(),
                 "criterion_state":criterion.state_dict(),
                 "optimizer_state": optimizer.state_dict(),"best_loss": best_loss}
        save_path,_,_,time_stamp = get_save_path(cfg,best_loss)
        torch.save(state, save_path)
        deleted_old = delete_old_checkpoint(save_path)
        print("Saving checkpoint '{}_best-loss_{}.pkl' (epoch {})".format(time_stamp,best_loss,epoch+1))
        if deleted_old:
            print('Deleted old checkpoints')
        plateau_count = 0
    else:
        plateau_count +=1
        
    return state,best_loss,plateau_count

def delete_old_checkpoint(save_path):
    removed = False
    path = '/'.join(save_path.split('/')[:-1])
    list_of_models = get_checkpoint_list(path)
    for model in list_of_models:
        if model not in save_path:
            os.remove(model)
            removed = True
    return removed
            
    

def stop_training(patience,plateau_count,early_stop,epoch,state):
    if plateau_count == patience and early_stop:
        print('Early Stopping after {} epochs: Patience of {} epochs reached.'.format(epoch+1,plateau_count))
        print('Best Checkpoint:')
        return True

def get_color_inst(inst_seg):
    colour_inst = np.zeros((inst_seg.shape[0], inst_seg.shape[1], 3))
    colour_inst[:,:,:2] = inst_seg
    colour_inst = colour_inst - np.min(colour_inst)
    colour_inst = colour_inst / np.max(colour_inst)
    
    return colour_inst

def get_inst_img_from_contours(mask, seg_img, contours):
    diff = np.zeros_like(seg_img)
    for i in range(3):
        diff[:,:,i] = seg_img[:,:,i]*(1-contours)*mask
    
    _, labels = cv2.connectedComponents(diff[:,:,2])
    instance_img = to_rgb(labels)
    return instance_img

def get_pan_img(mask, seg_img, inst_img):
    for i in range(3):
        seg_img[:,:,i] = seg_img[:,:,i]*(1-mask)

    return seg_img + inst_img

def add_images_to_writer(inputs,outputs,predictions,targets,writer,task,epoch,train=False):
    
    state = 'train' if train else 'validation'
    img = inputs[0,:,:,:]
    writer.add_image('Images/{}/Input_image'.format(state),img,epoch,dataformats='CHW')
    
    if task == 'semantic':
        img = decode_segmap(predictions[task][0,:,:].detach().cpu())
        target = decode_segmap(targets[task][0,:,:].detach().cpu())
        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')
    
    elif task == 'instance_contour':
        img = decode_segmap(predictions[task][0,:,:].detach().cpu(),nc=11,labels=inst_labels)
        contours = img[:,:,0]
        contours[contours>0] = 1
        seg = predictions['semantic'][0,:,:].cpu().numpy()
        mask = seg>=11
        instance_img, panoptic_img = getPanopticFromContour(mask, seg, contours)
    
        target = decode_segmap(targets[task][0,:,:].cpu(),nc=11,labels=inst_labels)
        gt_contours = target[:,:,0]
        gt_contours[gt_contours>0] = 1
        gt_mask = targets['semantic'][0,:,:].cpu().numpy()>=11
        gt_seg_img = targets['semantic'][0,:,:].cpu().numpy()
        gt_instance_img, gt_panoptic_img  = getPanopticFromContour(gt_mask, gt_seg_img, gt_contours)

        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/gt/instance_seg_contour'.format(state,task),
                         to_rgb(gt_instance_img), epoch,dataformats='HWC')
        writer.add_image('Images/{}/gt/panoptic_seg_contour'.format(state,task),
                         to_rgb(gt_panoptic_img), epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/instance_seg_contour'.format(state,task),
                         to_rgb(instance_img), epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/panoptic_seg_contour'.format(state,task),
                         to_rgb(panoptic_img), epoch,dataformats='HWC')
    
       
    elif task == 'instance_regression':
        if train:
            vecs = outputs[task][0,:,:,:].detach().cpu().numpy()
            img = get_color_inst(vecs.transpose(1,2,0))
        else:
            vecs = outputs[task][0,:,:,:].cpu().numpy()
            img = get_color_inst(vecs.transpose(1,2,0))
        

        gt_vecs = targets[task][0,:,:,:].cpu().numpy()
        gt_img = get_color_inst(gt_vecs.transpose(1,2,0))
        

        writer.add_image('Images/{}/gt/{}_vecs'.format(state,task),gt_img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}_vecs'.format(state,task),img,epoch,dataformats='HWC')
        
        
    elif task == 'instance_heatmap':
        if train:
            img = outputs[task][0,:,:].detach().cpu().numpy()
        else:
            img = outputs[task][0,:,:].cpu().numpy()
        gt_img = targets[task][0,:,:].cpu().unsqueeze(0).numpy()
        writer.add_image('Images/{}/gt/{}'.format(state,task),gt_img,epoch)
        writer.add_image('Images/{}/det/{}'.format(state,task),np.clip(img,0,np.max(img)),epoch)
            
    elif task == 'instance_probs':
        if train:
            img = decode_segmap(predictions[task][0,:,:].detach().cpu(),nc=2,labels = prob_labels)
        else:
            img = decode_segmap(predictions[task][0,:,:].cpu(),nc=2,labels = prob_labels)
        target = decode_segmap(targets[task][0,:,:].cpu(),nc=2,labels = prob_labels)
        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')
        
    
    elif task == 'disparity':
        if train:
            img = predictions[task][0,0,:,:].detach().cpu().unsqueeze(0).numpy().astype(np.uint8)
        else:
            img = predictions[task][0,0,:,:].cpu().unsqueeze(0).numpy().astype(np.uint8)
        gt_img = targets[task][0,0,:,:].cpu().unsqueeze(0).numpy().astype(np.uint8)
        writer.add_image('Images/{}/gt/{}'.format(state,task),gt_img,epoch)
        writer.add_image('Images/{}/det/{}'.format(state,task),img,epoch)
