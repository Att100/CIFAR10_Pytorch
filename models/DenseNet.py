import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(inCH)
        self.conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH,
                         out_channels=outCH, stride=stride, padding=padding)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2d(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, inCH, k=32):
        super(BottleNeck, self).__init__()
        self.conv2d_1x1 = BasicConv2d(ksize=1, inCH=inCH, outCH=4*k)
        self.conv2d_3x3 = BasicConv2d(ksize=3, inCH=4*k, outCH=k, padding=1)

    def forward(self, x):
        left = self.conv2d_1x1(x)
        left = self.conv2d_3x3(left)
        out = torch.cat([x, left], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, inCH, layernum=6, k=32):
        super(DenseBlock, self).__init__()
        self.layernum = layernum
        self.k = k
        self.inCH = inCH
        self.outCH = inCH + k * layernum
        self.block = self.make_layer(layernum)

    def forward(self, x):
        out = self.block(x)
        return out

    def make_layer(self, layernum):
        layers = []
        inchannels = self.inCH
        for i in range(layernum):
            layers.append(BottleNeck(inCH=inchannels, k=self.k))
            inchannels += self.k
        return nn.Sequential(*layers)


class Transition(nn.Module):
    def __init__(self, inCH, theta=0.5):
        super(Transition, self).__init__()
        self.outCH = int(math.floor(theta*inCH))
        self.bn = nn.BatchNorm2d(inCH)
        self.conv2d_1x1 = nn.Conv2d(kernel_size=1, in_channels=inCH, out_channels=self.outCH)
        self.avgpool = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2d_1x1(x)
        x = self.avgpool(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, k=32, theta=0.5):
        super(DenseNet121 ,self).__init__()
        self.k=k
        self.theta = theta
        self.pre_layer = BasicConv2d(ksize=3, inCH=3, outCH=2*self.k, padding=1)
        self.DenseBlock_1 = DenseBlock(inCH=2*self.k, layernum=6, k=self.k)
        self.Transition_1 = Transition(inCH=self.DenseBlock_1.outCH, theta=self.theta)
        self.DenseBlock_2 = DenseBlock(inCH=self.Transition_1.outCH, layernum=12, k=self.k)
        self.Transition_2 = Transition(inCH=self.DenseBlock_2.outCH, theta=self.theta)
        self.DenseBlock_3 = DenseBlock(inCH=self.Transition_2.outCH, layernum=24, k=self.k)
        self.Transition_3 = Transition(inCH=self.DenseBlock_3.outCH, theta=self.theta)
        self.DenseBlock_4 = DenseBlock(inCH=self.Transition_3.outCH, layernum=16, k=self.k)
        self.bn = nn.BatchNorm2d(self.DenseBlock_4.outCH)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.linear = nn.Linear(self.DenseBlock_4.outCH*1*1, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.DenseBlock_1(x)
        x = self.Transition_1(x)
        x = self.DenseBlock_2(x)
        x = self.Transition_2(x)
        x = self.DenseBlock_3(x)
        x = self.Transition_3(x)
        x = self.DenseBlock_4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, self.DenseBlock_4.outCH*1*1)
        x = self.linear(x)
        return x