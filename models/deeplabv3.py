from torch import nn
from torch.nn import functional as F
import torch


class DeepLabv3(nn.Module):
    def __init__(self, in_planes=2048, out_planes=[256],
                 n_class=19, atrous_rates=[12, 24, 36],
                 activation='ReLU', activate_last=True):
        super().__init__()
        out_planes = out_planes[0]
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_planes, out_planes, rate))

        modules.append(ASPPPooling(in_planes, out_planes))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.classifier = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        score = self.classifier(res)
        return score, res


class ASPPPooling(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear',
                             align_corners=False)


class ASPPConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dilation):
        modules = [
            nn.Conv2d(in_planes, out_planes, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
