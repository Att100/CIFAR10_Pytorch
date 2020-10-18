import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, inCH, outCh, ksize, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                    out_channels=outCh, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outCh)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_1_1 = BasicConv2d(3, 64, 3, padding=(1, 1))
        self.conv_1_2 = BasicConv2d(64, 64, 3, padding=(1, 1))
        self.maxpool_1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_2_1 = BasicConv2d(64, 128, 3, padding=(1, 1))
        self.conv_2_2 = BasicConv2d(128, 128, 3, padding=(1, 1))
        self.maxpool_2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_3_1 = BasicConv2d(128, 256, 3, padding=(1, 1))
        self.conv_3_2 = BasicConv2d(256, 256, 3, padding=(1, 1))
        self.conv_3_3 = BasicConv2d(256, 256, 3, padding=(1, 1))
        self.maxpool_3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_4_1 = BasicConv2d(256, 512, 3, padding=(1, 1))
        self.conv_4_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_4_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.maxpool_4 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_5_1 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_5_2 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.conv_5_3 = BasicConv2d(512, 512, 3, padding=(1, 1))
        self.maxpool_5 = nn.MaxPool2d((2, 2), stride=2)

        self.fc_1 = nn.Linear(512*1*1, 256)
        self.dp_1 = nn.Dropout()
        self.fc_2 = nn.Linear(256, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.fc_3 = nn.Linear(128, 10)

    def forward(self, x):

        x = F.relu(self.conv_1_1(x))
        x = self.conv_1_2(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = F.relu(self.conv_2_1(x))
        x = self.conv_2_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)

        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = self.conv_3_3(x)
        x = F.relu(x)
        x = self.maxpool_3(x)

        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = self.conv_4_3(x)
        x = F.relu(x)
        x = self.maxpool_4(x)

        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = self.conv_5_3(x)
        x = F.relu(x)
        x = self.maxpool_5(x)

        x = x.view(-1, 512*1*1)
        x = F.relu(self.fc_1(x))
        x = self.dp_1(x)
        x = F.relu(self.fc_2(x))
        x = self.bn_1(x)
        x = self.fc_3(x)
        return x