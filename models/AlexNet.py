import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv2d_1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=96, padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.maxpool_1 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Conv2d_2 = nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2)
        self.bn_2 = nn.BatchNorm2d(256)
        self.maxpool_2 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Conv2d_3 = nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1)
        self.Conv2d_4 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1)
        self.Conv2d_5 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.maxpool_3 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, 2048)
        self.dp_1 = nn.Dropout()
        self.fc_2 = nn.Linear(2048, 1024)
        self.dp_2 = nn.Dropout()
        self.fc_3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.Conv2d_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)

        x = F.relu(self.Conv2d_3(x))
        x = F.relu(self.Conv2d_4(x))
        x = F.relu(self.Conv2d_5(x))
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.maxpool_3(x)

        x = x.view(-1, 4*4*256)
        x = F.relu(self.fc_1(x))
        x = self.dp_1(x)
        x = F.relu(self.fc_2(x))
        x = self.dp_2(x)
        x = self.fc_3(x)
        return x