#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import argparse
from tqdm import tqdm

from utils.data_utils import get_weights, get_cfg
from utils.loss_utils import averageMeter
from utils.dataloader import get_dataloaders
from utils.train_utils import get_device, get_config_params, get_model, \
                              init_optimizer, get_losses_and_metrics, \
                              if_checkpoint_exists, load_checkpoint, \
                              train_step, validation_step, print_metrics, \
                              stop_training, save_model


def train(cfg):
    # Define Configuration parameters
    weights = get_weights(cfg)
    device = get_device(cfg)
    params, epochs, patience, early_stop, base_dir, exp_name, \
    resume_training, print_interval, best_loss, start_iter, \
    plateau_count, state = get_config_params(cfg)

    # Define dataloaders, model, optimizers and metrics
    dataloaders = get_dataloaders(cfg)
    model = get_model(cfg, device)
    optimizer = init_optimizer(model, params)
    train_loss_meters, val_loss_meters, val_metrics = get_losses_and_metrics(cfg)
    
    check_point_exists = if_checkpoint_exists(exp_name,base_dir)
    if check_point_exists and resume_training:
        model,optimizer,start_iter,best_loss = load_checkpoint(model,optimizer,
                                                               exp_name,base_dir)
    else:
        print("Begining Training from Scratch")
            
    for epoch in range(epochs):
        print('\n********************** Epoch {} **********************'.format(epoch+1))
        print('********************** Training *********************')        
        running_loss = averageMeter()
        running_val_loss = averageMeter()
        n_steps = len(dataloaders['train'])        
        for step, data in tqdm(enumerate(dataloaders['train'])):
            train_step(model,data,optimizer,cfg,device,weights,running_loss,
                       train_loss_meters,print_interval,n_steps,epoch,step)            
        val_metrics, current_loss = validation_step(model,dataloaders,cfg,device,
                                                    weights,running_val_loss,
                                                    val_loss_meters,val_metrics,epoch)
        print_metrics(val_metrics)
        state,best_loss,plateau_count = save_model(model,optimizer,cfg,current_loss,
                                                   best_loss,plateau_count,
                                                   start_iter,epoch,state)             
        stop_training(patience,plateau_count,early_stop,epoch,state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",nargs="?",type=str,default="configs/base.yml",help="Configuration to use")
    args = parser.parse_args()    
    cfg = get_cfg(args.config)
    train(cfg)