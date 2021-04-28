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
torch.manual_seed(42)
random.seed(42)

from dataloaders import dataLoader, Train, Test
from model import Net, LinearRegression

import wandb
wandb.login()
net = LinearRegression()
o1 = optim.Adam(net.parameters(), 0.01, weight_decay=0.001)
optim1 = optim.Adam(net.parameters(), 0.1, weight_decay=0.001)

o2 = optim.Adam(net.parameters(), 0.001, weight_decay=0.001)
o3 = optim.Adam(net.parameters(), 0.00001, weight_decay=0.001)
optims = {
    10: o1, 
    22: o2,
    31:o3

    }
crit = nn.MSELoss()
wandb.init(project="CRISPR POTS", config={
    "learning_rate": [0.1, 0.01, 0.001, 0.00001],
    "optim": "ADAM",
    "loss": "MSE",
    "architecture": "Linear-Model",
    "dataset": "CRSIPRSQL",
    "epochs": 1000,
    "batch": 256
})

print('loading data')
data = dataLoader(batch=256)
wandb.watch(net)
Train(50, optim1, crit, 11, data[0], data[1], net, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), optims, "linearRegression.txt")
PATH = f'.linear_net.pth'
torch.save(net.state_dict(), PATH)
print(f"Net saved to {PATH}")
Test(net, data[1][:5], torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), crit, "linearRegression.txt")

net.eval()
correct = 0
total = 0
totalloss = 0
loss = 0
# with torch.no_grad():
#     for i, d in enumerate(data[1][:5], 0):        
#         inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#         outputs = net(inputs)
#         print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")

   