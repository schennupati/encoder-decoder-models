from torch import nn
from utils.model_uitls import get_activation
# TODO: Update FPN to take Cross-Stitch


class _FPNModel(nn.Module):

    def __init__(self, in_planes=[2048, 1024, 512], out_planes=[256, 128],
                 n_class=19, activation='ReLU', activate_last=True):
        super().__init__()
        self.out_planes = out_planes
        self.n_class = n_class
        self.upsample1_1 = self.upsample(
            in_planes[0], out_planes[0], activation)
        self.upsample1_2 = self.upsample(
            out_planes[0], out_planes[0], activation)
        self.upsample1_3 = self.upsample(
            out_planes[0], out_planes[1], activation)
        self.upsample2_1 = self.upsample(
            in_planes[1], out_planes[0], activation)
        self.upsample2_2 = self.upsample(
            out_planes[0], out_planes[1], activation)
        self.upsample3 = self.upsample(in_planes[2], out_planes[1], activation)
        self.upsample4 = self.upsample(in_planes[3], out_planes[1], activation,
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
        feats = [feat1, feat2, feat3, feat4]

        return score, feats

    def upsample(self, in_channels, out_channels, activation='ReLU', activate_last=True,
                 upsample=True, kernel_size=3, factor=2, dilation=1):

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, padding=int(kernel_size/2)))
        layers.append(nn.BatchNorm2d(out_channels))
        if activate_last:
            layers.append(get_activation(activation, True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=factor,
                                      mode='bilinear', align_corners=False))

        return nn.Sequential(*layers)


class FPN(_FPNModel):
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
