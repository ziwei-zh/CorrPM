import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from networks.non_local import NONLocalBlock2D


class NL_subnet(nn.Module):
    def __init__(self, inplanes=2048, planes=128):
        super(NL_subnet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.NL = NONLocalBlock2D(in_channels=planes)
        self.conv2 = nn.Conv2d(planes, 1, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.NL(x)
        x = self.conv2(x)

        return x

class Nonlocal_correlation(nn.Module):
    def __init__(self, conv_planes=768, out_planes=20, mid_channels=256):
        super(Nonlocal_correlation, self).__init__()
        self.conv_planes = conv_planes
        self.out_planes = out_planes
        self.mid_channels = mid_channels

        self.conv_layer = nn.Conv2d(conv_planes, mid_channels, kernel_size=1, padding=0, bias=False)
        self.nonLocal = NONLocalBlock2D(in_channels=mid_channels)
        self.out_layer = nn.Conv2d(mid_channels, out_planes, kernel_size=1, padding=0, bias=True)

    def forward(self, x1, x2): # x, edge_fea
        if x1.size(1) == x2.size(1): # edge_fea, pose_fea
            x1 = self.conv_layer(x1)

        x2_ = self.conv_layer(x2)
        relation = self.nonLocal(x1, x2_)
        out = self.out_layer(relation)

        return out


class Nonlocal_psp_correlation(nn.Module):
    def __init__(self, conv_planes=768, out_planes=20, mid_channels=256):
        super(Nonlocal_correlation, self).__init__()
        self.conv_planes = conv_planes
        self.out_planes = out_planes
        self.mid_channels = mid_channels

        self.conv_layer = PSPModule(conv_planes, mid_channels)
        self.nonLocal = NONLocalBlock2D(in_channels=mid_channels)
        self.out_layer = nn.Conv2d(mid_channels, out_planes, kernel_size=1, padding=0, bias=True)

    def forward(self, x1, x2): # x, edge_fea
        if x1.size(1) == x2.size(1): # edge_fea, pose_fea
            x1 = self.conv_layer(x1)

        x2_ = self.conv_layer(x2)
        relation = self.nonLocal(x1, x2_)
        out = self.out_layer(relation)

        return out



class NONLocal(nn.Module):
    def __init__(self, inplanes=2048, planes=128):
        super(NONLocal, self).__init__()
        self.NL0 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL1 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL2 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL3 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL4 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL5 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL6 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL7 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL8 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL9 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL10 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL11 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL12 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL13 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL14 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL15 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL16 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL17 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL18 = NL_subnet(inplanes=inplanes, planes=planes)
        self.NL19 = NL_subnet(inplanes=inplanes, planes=planes)

    def forward(self, x):
        x0 = self.NL0(x)
        x1 = self.NL1(x)
        x2 = self.NL2(x)
        x3 = self.NL3(x)
        x4 = self.NL4(x)
        x5 = self.NL5(x)
        x6 = self.NL6(x)
        x7 = self.NL7(x)
        x8 = self.NL8(x)
        x9 = self.NL9(x)
        x10 = self.NL10(x)
        x11 = self.NL11(x)
        x12 = self.NL12(x)
        x13 = self.NL13(x)
        x14 = self.NL14(x)
        x15 = self.NL15(x)
        x16 = self.NL16(x)
        x17 = self.NL17(x)
        x18 = self.NL18(x)
        x19 = self.NL19(x)
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19], dim=1)

        return x
