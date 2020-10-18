import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                        out_channels=outCH, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(outCH)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(DepthwiseConv2d, self).__init__()
        self.dwConv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                            out_channels=inCH, stride=stride, padding=padding, groups=inCH)
        self.bn = nn.BatchNorm2d(inCH)
        self.relu = nn.ReLU(inplace=True)
        self.pointwiseConv2d = BasicConv2d(ksize=1, inCH=inCH, outCH=outCH)

    def forward(self, x):
        x = self.dwConv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwiseConv2d(x)
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.pre_layer = BasicConv2d(ksize=3, inCH=3, outCH=32)
        self.Depthwise = nn.Sequential(
            DepthwiseConv2d(ksize=3, inCH=32, outCH=64, padding=1),
            DepthwiseConv2d(ksize=3, inCH=64, outCH=128, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=128, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=256, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=256, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=512, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=1024, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=1024, outCH=1024, padding=1)
        )
        self.avgpool = nn.AvgPool2d((4, 4))
        self.linear = nn.Linear(1024*1*1, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.Depthwise(x)
        x = self.avgpool(x)
        x = x.view(-1, 1*1*1024)
        x = self.linear(x)
        return x