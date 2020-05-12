#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:47:42 2019

@author: sumche
"""

import math

import mlconfig
import torch
from torch import nn

from .utils import load_state_dict_from_url

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size,
                       stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


@mlconfig.register
class EfficientNet(nn.Module):

    def __init__(self, multiscale=False, width_mult=1.0, depth_mult=1.0,
                 dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
        self.multiscale = multiscale
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.dropout_rate = dropout_rate
        self.in_planes_map = {}

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        # Layer1 outputs downsized scale 2
        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]
        self.layer1 = nn.Sequential(*features)
        self.in_channels = out_channels
        self.in_planes_map[0] = self.in_channels

        # Layer 2 outputs downsized scale 4
        features = self._make_layer(settings[0])
        features += self._make_layer(settings[1])
        self.layer2 = nn.Sequential(*features)
        self.in_planes_map[1] = self.in_channels

        # Layer 3 outputs downsized scale 8
        features = self._make_layer(settings[2])
        self.layer3 = nn.Sequential(*features)
        self.in_planes_map[2] = self.in_channels

        # Layer 4 outputs downsized scale 16
        features = self._make_layer(settings[3])
        self.layer4 = nn.Sequential(*features)
        self.in_planes_map[3] = self.in_channels

        # Layer 5 outputs downsized scale 32
        features = self._make_layer(settings[4])
        features += self._make_layer(settings[5])
        self.layer5 = nn.Sequential(*features)
        self.in_planes_map[4] = self.in_channels

        # Layer 6 continues to maintain scale 32
        features = self._make_layer(settings[6])
        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(self.in_channels, last_channels, 1)]
        self.last_layer = nn.Sequential(*features)

        self.classifier = nn.Sequential(nn.Dropout(dropout_rate),
                                        nn.Linear(last_channels, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, setting):
        t, c, n, s, k = setting
        features = []
        out_channels = _round_filters(c, self.width_mult)
        repeats = _round_repeats(n, self.depth_mult)
        for i in range(repeats):
            stride = s if i == 0 else 1
            features += [MBConvBlock(self.in_channels, out_channels,
                                     expand_ratio=t, stride=stride,
                                     kernel_size=k)]
            self.in_channels = out_channels

        return features

    def forward(self, x):
        down_sampled_2 = self.layer1(x)
        down_sampled_4 = self.layer2(down_sampled_2)
        down_sampled_8 = self.layer3(down_sampled_4)
        down_sampled_16 = self.layer4(down_sampled_8)
        down_sampled_32 = self.layer5(down_sampled_16)
        out = self.last_layer(down_sampled_32)
        out = out.mean([2, 3])
        out = self.classifier(out)
        intermed = [down_sampled_2,
                    down_sampled_4,
                    down_sampled_8,
                    down_sampled_16,
                    down_sampled_32]
        if self.multiscale:
            return out, intermed
        else:
            return out


def _efficientnet(arch, pretrained, progress, multiscale=False, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(multiscale, width_mult, depth_mult,
                         dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


@mlconfig.register
def efficientnet_b0(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b1(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b1', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b2(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b2', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b3(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b4(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b4', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b5(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b5', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b6(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b6', pretrained,
                         progress, multiscale, **kwargs)


@mlconfig.register
def efficientnet_b7(pretrained=False, progress=True,
                    multiscale=False, **kwargs):
    return _efficientnet('efficientnet_b7', pretrained,
                         progress, multiscale, **kwargs)
