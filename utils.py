import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torch.optim as optim
import matplotlib.pyplot as plt 
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import time
import math
import cv2


def getDataloader(root="./dataset", batchsize=256, worker=2):
    transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root=root, train=True,download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batchsize,shuffle=True, num_workers=worker)
    testset = datasets.CIFAR10(root=root, train=False,download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batchsize,shuffle=False, num_workers=worker)
    return train_loader, test_loader


def retrieveCurve(path, state='train', kind='acc'):
    path = path + "run-.-tag-" + state + "_" + kind + ".csv"
    f = open(path, 'r')
    reader = csv.reader(f)
    data = list(reader)
    curve = []
    for i in range(1, len(data)):
        curve.append(float(data[i][2]))
    return curve


def graphCurve(accuracy={'net':[1.0, 1.0, 1.0]}, xlab='epoch', ylab="accuracy (%)",
            bbox=(0.85, 0.15), save=False, saveName="untitled.jpg", dpi=800):
    fig = plt.figure(2)
    for key in list(accuracy.keys()):
        lb = key
        data = accuracy[key]
        plt.plot(data, label=lb)
    y_major_locator=MultipleLocator(10)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(ls='--')
    plt.xlim((0, 149))
    plt.ylim((50, 100))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(' ')
    fig.legend(loc=4, bbox_to_anchor=bbox)
    if save:
        plt.savefig(saveName, dpi=dpi)
    plt.show()


class ProgressBar:
    def __init__(self, maxStep=150, fill="#"):
        self.maxStep = maxStep
        self.fill = fill
        self.barLength = 20
        self.barInterval = 0
        self.prInterval = 0
        self.count = 0
        self.progress = 0
        self.barlenSmaller = True
        self.genBarCfg()

    def genBarCfg(self):
        if self.maxStep >= self.barLength:
            self.barInterval = math.ceil(self.maxStep / self.barLength)
        else:
            self.barlenSmaller = False
            self.barInterval = math.floor(self.barLength / self.maxStep)
        self.prInterval = 100 / self.maxStep

    def resetBar(self):
        self.count = 0
        self.progress = 0

    def updateBar(self, step, headData={'head':10}, endData={'end_1':2.2, 'end_2':1.0}, keep=False):
        head_str = "\r"
        end_str = " "
        process = ""
        if self.barlenSmaller:
            if step != 0 and step % self.barInterval == 0:
                self.count += 1
        else:
            self.count += self.barInterval
        self.progress += self.prInterval
        for key in headData.keys():
            head_str = head_str + key + ": " + str(headData[key]) + " "
        for key in endData.keys():
            end_str = end_str + key + ": " + str(endData[key]) + " "
        if step == self.maxStep:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (100.0, self.fill * self.barLength)
            process += end_str
            if not keep:
                process += "\n"
        else:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (round(self.progress, 1), self.fill * self.count)
            process += end_str
        print(process, end='', flush=True)


if __name__ == "__main__":

    bar = ProgressBar(maxStep=150)
    for epoch in range(50):
        for step in range(150):
            time.sleep(0.1)
            bar.updateBar(step+1, headData={}, endData={})
        bar.resetBar()

        

