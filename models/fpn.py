from torch import nn
from utils.model_uitls import get_activation
# TODO: Update FPN to take Cross-Stitch


class FPNModelSymmetric(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 activation='ReLU', activate_last=True):
        super().__init__()
        self.out_planes = out_planes
        self.upsample1_1 = upsample(
            in_planes[0], out_planes[0], activation)
        self.upsample1_2 = upsample(
            out_planes[0], out_planes[0], activation)
        self.upsample1_3 = upsample(
            out_planes[0], out_planes[1], activation)
        self.upsample2_1 = upsample(
            in_planes[1], out_planes[0], activation)
        self.upsample2_2 = upsample(
            out_planes[0], out_planes[1], activation)
        self.upsample3 = upsample(in_planes[2], out_planes[1], activation)
        self.upsample4 = upsample(in_planes[3], out_planes[1], activation,
                                  upsample=False, kernel_size=3)

    def forward(self, intermediate_result):
        feat1 = self.upsample1_3(self.upsample1_2(
            self.upsample1_1(intermediate_result[-1])))
        # size=(N, 512, x.H/16, x.W/16)
        feat2 = self.upsample2_2(self.upsample2_1(intermediate_result[-2]))
        # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.upsample3(intermediate_result[-3])
        # size=(N, 128, x.H/4, x.W/4)
        feat4 = self.upsample4(intermediate_result[-4])
        score = feat1 + feat2 + feat3 + feat4
        feats = [feat4, feat3, feat2, feat1]
        return score, feats


class FPNModeltopdown(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 activation='ReLU', activate_last=True):
        super().__init__()
        self.out_planes = out_planes
        self.lateral1 = lateral(in_planes[0], out_planes[1], activation)
        self.lateral2 = lateral(in_planes[1], out_planes[1], activation)
        self.lateral3 = lateral(in_planes[2], out_planes[1], activation)
        self.lateral4 = lateral(in_planes[3], out_planes[1], activation)
        self.upsample1 = upsample(
            out_planes[1], out_planes[1], activation, conv=False)
        self.upsample2 = upsample(
            out_planes[1], out_planes[1], activation, conv=False)
        self.upsample3 = upsample(
            out_planes[1], out_planes[1], activation, conv=False)

    def forward(self, intermediate_result):
        feat1 = self.upsample1(self.lateral1(intermediate_result[-1]))
        # size=(N, 512, x.H/16, x.W/16)
        lat = self.lateral2(intermediate_result[-2])
        feat2 = self.upsample2(lat + feat1)
        # size=(N, 256, x.H/8, x.W/8)
        feat3 = self.upsample3(self.lateral3(intermediate_result[-3]) + feat2)
        # size=(N, 128, x.H/4, x.W/4)
        feats = [feat3, feat2, feat1]

        return feat3, feats


def upsample(in_channels, out_channels, activation='ReLU',
             activate_last=True, upsample=True, conv=True,
             kernel_size=3, factor=2, dilation=1):

    layers = []
    if conv:
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, padding=int(kernel_size/2)))
        layers.append(nn.BatchNorm2d(out_channels))
        # TODO: Add BatchNorm
        if activate_last:
            layers.append(get_activation(activation, True))
    if upsample:
        layers.append(nn.Upsample(scale_factor=factor,
                                  mode='bilinear', align_corners=False))

    return nn.Sequential(*layers)


def lateral(in_channels, out_channels, activation='ReLU',
            activate_last=True, kernel_size=3, factor=2, dilation=1):

    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels,
                            kernel_size, padding=int(kernel_size/2)))
    layers.append(nn.BatchNorm2d(out_channels))
    # TODO: Add BatchNorm
    if activate_last:
        layers.append(get_activation(activation, True))

    return nn.Sequential(*layers)


def build_fpn(name, in_channles, out_channels, *kwargs):
    if name == 'top_down':
        return FPNModeltopdown(in_channles, out_channels, *kwargs)
    elif name == 'symmetric':
        return FPNModelSymmetric(in_channles, out_channels, *kwargs)
