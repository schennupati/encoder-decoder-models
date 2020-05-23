# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from torch import nn



def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": lambda channels: nn.BatchNorm2d(channels),
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
        }[norm]
    return norm(out_channels)
