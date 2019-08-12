#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:50:59 2019

@author: sumche
"""
import os
import shutil

path_to_additional_data = '/home/sumche/datasets/Cityscapes/gtFine'
path_to_annotations = '/home/sumche/datasets/Cityscapes/gtFine'

def get_total_files_count(path,ext='.png'):
    count = 0
    for root, dirs, names in os.walk(path, topdown=False):
        for name in names:
            if name.endswith(ext):
                count += 1 
    return count

new_data_count = get_total_files_count(path_to_additional_data)
annotations_count = get_total_files_count(path_to_annotations,'color.png')

print(new_data_count,annotations_count) 

if new_data_count == annotations_count:
    for root, dirs, names in os.walk(path_to_additional_data, topdown=False):
        for name in names:
            if name.endswith(".jpeg") or name.endswith(".json") or name.endswith(".png"):
                src = os.path.join(path_to_additional_data,
                                   root.split('/')[-2], 
                                   root.split('/')[-1],
                                   name)
                dest = os.path.join(path_to_annotations,
                                    root.split('/')[-2], 
                                    root.split('/')[-1],
                                    name)
                shutil.copy2(src,dest,follow_symlinks=False)
                

                

    