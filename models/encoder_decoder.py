#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:00:48 2019

@author: sumche
"""
#TODO: Update to use VGG, DenseNet etc  
#TODO: Update to use PSPNet, DeepLabV3 etc as decoders
#TODO: Implement multi-task feature.

from collections import OrderedDict

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.fcn import FCN 
from models.fpn import FPN
from models.vgg import VGGNet

__all__ = ['FCN']

inplanes_map = {
                'resnet':
                         {
                          'resnet18':[512,256,128],'resnet34':[512,256,128],
                          'resnet50':[2048,1024,512],'resnet101':[2048,1024,512],
                          'resnet152':[2048,1024,512],'resnext50_32x4d':[2048,1024,512],
                          'resnext101_32x8d':[2048,1024,512],'wide_resnet50_2':[2048,1024,512],
                          'wide_resnet101_2':[2048,1024,512]
                          },
                'vgg':
                      {
                       'vgg16':[512,512,256]
                       }
                }
    
decoder_map = {'fcn': (FCN),'fpn':(FPN)}

def get_encoder_decoder(cfg, pretrained_backbone=True):
    
    encoder_name = cfg['model']['encoder']
    decoder_name = cfg['model']['decoder']
    tasks = cfg['tasks']
    
    if 'resnet' in encoder_name:
        if decoder_name == 'fpn':
            encoder    = resnet_fpn_backbone(encoder_name,pretrained=True)
            inplanes   = [256,256,156]
            decoder_fn = decoder_map['fpn']
            
        elif decoder_name == 'fcn':
            return_layers = {'layer2': 'out_1','layer3': 'out_2','layer4': 'out_final'}
            encoder = resnet.__dict__[encoder_name](pretrained=pretrained_backbone,
                                 replace_stride_with_dilation=[False, False, False])       
            encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
            inplanes = inplanes_map['resnet'][encoder_name]
            decoder_fn = decoder_map['fcn']
        
    elif 'vgg' in encoder_name:
        encoder = VGGNet(model=encoder_name, requires_grad=True)
        inplanes = inplanes_map['vgg'][encoder_name]
        if decoder_name =='fpn':
            decoder_fn = decoder_map['fpn']
        elif decoder_name =='fcn':
            decoder_fn = decoder_map['fcn']
    decoders = nn.ModuleList()
    
    for task in tasks.keys():
        decoder = decoder_fn(inplanes, tasks[task]["classes"])
        decoders.extend([decoder])
    model = Encoder_Decoder(encoder, decoders)
    
    return model


class _Encoder_Decoder(nn.Module):

    def __init__(self, encoder, decoders):
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders

    def forward(self, x):
        output = self.encoder(x)
        intermediate_result = OrderedDict()
        layers = [k for k,_ in output.items()]
        for layer in layers:
            intermediate_result[layer] = output[layer]
        outputs = []
        
        for decoder in self.decoders:
            outputs.append(decoder(intermediate_result,layers))
        
        return outputs  # size=(N, n_class, x.H/1, x.W/1)

class Encoder_Decoder(_Encoder_Decoder):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass
