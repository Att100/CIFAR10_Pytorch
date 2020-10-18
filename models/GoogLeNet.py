import torch
import torch.nn as nn
import torch.nn.functional as F


InceptionConfig = {
    '3a':{'1x1':64, 'r3x3':96, '3x3':128, 'r5x5':16, '5x5':32, 'pool':32},
    '3b':{'1x1':128, 'r3x3':128, '3x3':192, 'r5x5':32, '5x5':96, 'pool':64},
    '4a':{'1x1':192, 'r3x3':96, '3x3':208, 'r5x5':16, '5x5':48, 'pool':64},
    '4b':{'1x1':160, 'r3x3':112, '3x3':224, 'r5x5':24, '5x5':64, 'pool':64},
    '4c':{'1x1':128, 'r3x3':128, '3x3':256, 'r5x5':24, '5x5':64, 'pool':64},
    '4d':{'1x1':112, 'r3x3':144, '3x3':288, 'r5x5':32, '5x5':64, 'pool':64},
    '4e':{'1x1':256, 'r3x3':160, '3x3':320, 'r5x5':32, '5x5':128, 'pool':128},
    '5a':{'1x1':256, 'r3x3':160, '3x3':320, 'r5x5':32, '5x5':128, 'pool':128},
    '5b':{'1x1':384, 'r3x3':192, '3x3':384, 'r5x5':48, '5x5':128, 'pool':128},
}


class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCh, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                    out_channels=outCh, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outCh)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, inCH, config, name):
        super(Inception, self).__init__()
        self.cfg = config[name]
        self.branch_1 = BasicConv2d(1, inCH, self.cfg['1x1'])
        self.branch_2 = nn.Sequential(
            BasicConv2d(1, inCH, self.cfg['r3x3']),
            BasicConv2d(3, self.cfg['r3x3'], self.cfg['3x3'], padding=1)
        )
        # 5 x 5(x1) => 3 x 3(x2)  
        self.branch_3 = nn.Sequential(
            BasicConv2d(1, inCH, self.cfg['r5x5']),
            BasicConv2d(3, self.cfg['r5x5'], self.cfg['5x5'], padding=1),
            BasicConv2d(3, self.cfg['5x5'], self.cfg['5x5'], padding=1)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=1, padding=1),
            BasicConv2d(1, inCH, self.cfg['pool'])
        )

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out_4 = self.branch_4(x)
        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        return out


class AuxClassifier(nn.Module):
    def __init__(self, inCH):
        super(AuxClassifier, self).__init__()
        self.avgPool = nn.AvgPool2d((5, 5), stride=3)
        self.Conv2d = BasicConv2d(1, inCH, 128)
        self.fc1 = nn.Linear(4*4*128, 1024)
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.avgPool(x)
        x = self.Conv2d(x)
        x = x.view(-1, 4*4*128)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, config=InceptionConfig):
        super(GoogLeNet, self).__init__()
        self.cfg = config

        self.Conv2d_1 = BasicConv2d(ksize=3, inCH=3, outCh=64, padding=1)
        self.maxPool_1 = nn.MaxPool2d((3, 3), stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.Conv2d_2 = BasicConv2d(ksize=1, inCH=64, outCh=64)
        self.Conv2d_3 = BasicConv2d(ksize=3, inCH=64, outCh=192, padding=1)
        self.bn_2 = nn.BatchNorm2d(192)
        self.maxPool_2 = nn.MaxPool2d((3, 3), stride=1, padding=1)

        self.Inception_3a = Inception(inCH=192, config=self.cfg, name='3a')
        self.Inception_3b = Inception(inCH=256, config=self.cfg, name='3b')
        self.maxPool_3 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Inception_4a = Inception(inCH=480, config=self.cfg, name='4a')
        self.Inception_4b = Inception(inCH=512, config=self.cfg, name='4b')
        self.Inception_4c = Inception(inCH=512, config=self.cfg, name='4c')
        self.Inception_4d = Inception(inCH=512, config=self.cfg, name='4d')
        self.Inception_4e = Inception(inCH=528, config=self.cfg, name='4e')
        self.maxPool_4 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Inception_5a = Inception(inCH=832, config=self.cfg, name='5a')
        self.Inception_5b = Inception(inCH=832, config=self.cfg, name='5b')

        self.avgPool = nn.AvgPool2d((8, 8), stride=1)
        self.dp = nn.Dropout(0.4)
        self.fc = nn.Linear(1*1*1024, 10)

        self.auxClassifier_1 = AuxClassifier(inCH=512)
        self.auxClassifier_2 = AuxClassifier(inCH=528)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.maxPool_1(x)
        x = self.bn_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x = self.bn_2(x)
        x = self.maxPool_2(x)
        
        x = self.Inception_3a(x)
        x = self.Inception_3b(x)
        x = self.maxPool_3(x)

        x = self.Inception_4a(x)

        auxOut_1 = self.auxClassifier_1(x)
        x = self.Inception_4b(x)
        x = self.Inception_4c(x)
        x = self.Inception_4d(x)

        auxOut_2 = self.auxClassifier_2(x)
        x = self.Inception_4e(x)
        x = self.maxPool_4(x)

        x = self.Inception_5a(x)
        x = self.Inception_5b(x)

        x = self.avgPool(x)
        x = x.view(-1, 1*1*1024)
        x = self.dp(x)
        Out = self.fc(x)

        return Out , auxOut_1, auxOut_2