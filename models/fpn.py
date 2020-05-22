from torch import nn
from utils.model_uitls import get_activation, c2_xavier_fill
from models.layers import get_norm
# TODO: Update FPN to take Cross-Stitch


class FPNModelSymmetric(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 activation='ReLU', activate_last=True):
        super().__init__()
        self.out_planes = out_planes
        self.upsample1_1 = upsample(
            in_planes[0], out_planes[0], activation, activate_last)
        self.upsample1_2 = upsample(
            out_planes[0], out_planes[0], activation, activate_last)
        self.upsample1_3 = upsample(
            out_planes[0], out_planes[1], activation, activate_last)
        self.upsample2_1 = upsample(
            in_planes[1], out_planes[0], activation, activate_last)
        self.upsample2_2 = upsample(
            out_planes[0], out_planes[1], activation, activate_last)
        self.upsample3 = upsample(
            in_planes[2], out_planes[1], activation, activate_last)
        self.upsample4 = upsample(in_planes[3], out_planes[1], activation,
                                  activate_last, upsample=False, kernel_size=3)

    def forward(self, intermediate_result):
        x = self.upsample1_3(self.upsample1_2(
            self.upsample1_1(intermediate_result[-1])))
        # size=(N, 512, x.H/16, x.W/16)
        x = self.upsample2_2(self.upsample2_1(intermediate_result[-2])) + x
        # size=(N, 256, x.H/8, x.W/8)
        x = self.upsample3(intermediate_result[-3]) + x
        # size=(N, 128, x.H/4, x.W/4)
        x = self.upsample4(intermediate_result[-4]) + x
        # feats = [feat4, feat3, feat2, feat1]
        return x


class FPNModeltopdown(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 activation='ReLU', activate_last=True, multi_scale=False):
        super().__init__()
        self.out_planes = out_planes
        self.multi_scale = multi_scale
        self.lateral1 = lateral(in_planes[0], out_planes[1], activation)
        self.lateral2 = lateral(in_planes[1], out_planes[1], activation)
        self.lateral3 = lateral(in_planes[2], out_planes[1], activation)
        self.lateral4 = lateral(in_planes[3], out_planes[1], activation)
        self.upsample1 = upsample(
            out_planes[1], out_planes[1], activation, activate_last, conv=False)
        self.upsample2 = upsample(
            out_planes[1], out_planes[1], activation, activate_last, conv=False)
        self.upsample3 = upsample(
            out_planes[1], out_planes[1], activation, activate_last, conv=False)

    def forward(self, intermediate_result):
        if self.multi_scale:
            feat1 = self.upsample1(self.lateral1(intermediate_result[-1]))
            # size=(N, 512, x.H/16, x.W/16)
            lat = self.lateral2(intermediate_result[-2])
            feat2 = self.upsample2(lat + feat1)
            # size=(N, 256, x.H/8, x.W/8)
            feat3 = self.upsample3(self.lateral3(
                intermediate_result[-3]) + feat2)
            # size=(N, 128, x.H/4, x.W/4)
            feat4 = self.lateral4(intermediate_result[-4]) + feat3
            feats = [feat1, feat2, feat3, feat4]
            return feats
        else:
            x = self.upsample1(self.lateral1(intermediate_result[-1]))
            # size=(N, 512, x.H/16, x.W/16)
            x = self.upsample2(x + self.lateral2(intermediate_result[-2]))
            # size=(N, 256, x.H/8, x.W/8)
            x = self.upsample3(self.lateral3(
                intermediate_result[-3]) + x)
            # size=(N, 128, x.H/4, x.W/4)
            return x


class FPNModel(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 activation='ReLU', activate_last=True, multi_scale=False):
        super().__init__()
        self.out_planes = out_planes
        self.multi_scale = multi_scale
        out = out_planes[0]
        self.top_down = FPNModeltopdown(in_planes, [out, out], activation,
                                        activate_last=False, multi_scale=True)
        in_, out = out, out_planes[1]
        self.symmetric = FPNModelSymmetric([in_, in_, in_, in_],
                                           [out, out], activation,
                                           activate_last=True)
        if multi_scale:
            raise ValueError('FPN Model doesnot support multi-scale. \
                             Use topdown instead.')

    def forward(self, intermediate_result):
        return self.symmetric(self.top_down(intermediate_result))


def upsample(in_channels, out_channels, activation='ReLU',
             activate_last=True, upsample=True, conv=True,
             kernel_size=3, factor=2, dilation=1):

    layers = []
    if conv:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         padding=int(kernel_size/2), bias=False)
        c2_xavier_fill(conv)
        layers.append(conv)
        layers.append(get_norm('GN', out_channels))
        if activate_last:
            layers.append(get_activation(activation, True))

    if upsample:
        layers.append(nn.Upsample(scale_factor=2,
                                  mode="bilinear", align_corners=False))

    return nn.Sequential(*layers)


def lateral(in_channels, out_channels, activation='ReLU',
            activate_last=True, kernel_size=3, factor=2, dilation=1):

    layers = []
    conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size, padding=int(kernel_size/2), bias=False)
    c2_xavier_fill(conv)
    layers.append(conv)
    layers.append(get_norm('GN', out_channels))

    return nn.Sequential(*layers)


def build_fpn(in_channles, out_channels, name=None, *kwargs):
    if name == 'topdown':
        return FPNModeltopdown(in_channles, out_channels, *kwargs)
    elif name == 'symmetric':
        return FPNModelSymmetric(in_channles, out_channels, *kwargs)
    elif name is None:
        return FPNModel(in_channles, out_channels, *kwargs)
