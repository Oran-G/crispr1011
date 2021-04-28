# print("importing torch")

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
import numpy as np
from dataloaders import dataLoader, Test, compute_dataframe
from model import Net, LinearRegression
import pandas as pd
net = Net()

# print('Loading data.....')
# # data = dataLoader(batch=1)
# # crit = nn.MSELoss()
# PATH = f'logisticnet.pth'
# def get_param_count(model):
#     param_counts = [np.prod(p.size()) for p in model.parameters()]
#     return sum(param_counts)
# print(get_param_count(net))
# net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
# net.eval()
# df = pd.read_csv('crisprsql.csv')
# compute_dataframe(df=df, checkpoint_path=PATH).to_csv(logisticnet.csv)

print(torch.zeros((4, 23, 4))[0])

# correct = 0
# total = 0
# totalloss = 0
# loss = 0
# with torch.no_grad():
#     for i, d in enumerate(data[1][25:35], 0):        
#         inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#         outputs = net(inputs)
#         print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")

   