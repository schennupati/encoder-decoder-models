#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:59:23 2019

@author: sumche
"""
import torch
import time

from models.encoder_decoder import get_encoder_decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_name = 'fcn'
encoder_name = 'vgg16'

model = get_encoder_decoder(encoder_name, decoder_name, num_classes=21, fpn=True)
model = model.to(device)

print(model)
t = time.time()
out = model(torch.rand(1, 3, 512, 512))
elapsed = time.time() - t

#print(out.size)
print(elapsed)