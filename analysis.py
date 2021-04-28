print("importing torch")

import copy
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import os
import wandb
torch.manual_seed(42)
random.seed(42)
import numpy as np
from dataloaders import dataLoader, Test, NewTrain, rankDataLoader, roc
from model import Net, LinearRegression, LogisticRegression

if __name__ == '__main__':
   
    print('loading data')
    # dataLoader
    data, _ = rankDataLoader(batch=5)
    less = 0
    total = 0
    for i in data[1][50:60]:
        labels = i[1].flatten().tolist()
        for label in labels:
            if label <= .01:
                less+=1
            total+=1
    print(less / total)
    for i in data[1][50:60]:
        labels = i[1].flatten().tolist()
        print(labels)

    PATH = f'net.pth'
    net = Net(2)
    net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    for i in data[1][50:60]:
        print(net(i[0]).flatten())

    



    full = None

    for i in data[1][50:60]:
        if full == None:
            full = i[1]
        else:
            full = torch.cat([full, i[1]], 0)
    print(roc(
        full, torch.zeros((full.size(0), 1))
        )[.1][0])
