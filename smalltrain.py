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

from dataloaders import dataLoader, Test, NewTrain, rankDataLoader
from model import Net, LinearRegression, LogisticRegression
if __name__ == '__main__':
    wandb.login()
    net = Net(1)
    net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # LR Scedule I wanded momentum so I used SGD
    optim1 = optim.Adam(net.parameters(), 1, weight_decay=0.00125)

    optims = [
        # [50, optim.Adam(net.parameters(), 1,  weight_decay=0.00125)],
        # [50, optim.Adam(net.parameters(), 0.1, weight_decay=0.00125)],
        [50, optim.Adam(net.parameters(), 0.2, weight_decay=0.000125)], 
        [1500, optim.Adam(net.parameters(), 0.05, weight_decay=0.000125)], 
        # [200, optim.Adam(net.parameters(), 0.005, weight_decay=0.000125)]
        # [50, optim.Adam(net.parameters(), 0.00001)]
        ]
    crit = nn.MSELoss()
    wandb.init(project="crispr-pots", config={
        "learning_rate": 0.05,
        "optim": "ADAM",
        "loss": "MSE",
        "architecture": "RESNET 2",
        "dataset": "CRSIPRSQL validation point 64",
        "epochs": 1500,
        "batch": 25
    })
    wandb.watch(net)
    print('loading data')
    # dataLoader
    data, _ = rankDataLoader(batch=10)
    yhat = []
    ypred = []
    for batch in data[0][:4]:
        yhat.extend(batch[1].flatten().tolist())
        ypred.extend(net(batch[0]).flatten().tolist())
    
    # print(yhat, ypred)

    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    print(zip(yhat, ypred))
    print("pearsonr:", pearsonr(yhat, ypred))
    print('spearmanr:', spearmanr(yhat, ypred))
    pearson = [
        {'yhat': p1, 'ypred': p} for p1, p in zip(yhat, ypred)
    ]

    # look at selected predictions before training
    # with torch.no_grad():
    #     # for i, d in enumerate(data[1][:1], 0):   
    #     d = data[1][50]     
    #     inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #     outputs = net(inputs)
    #     print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")
    # train
    NewTrain(250, optim1, crit, 1000, data[0][:20], data[0][:20], net, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), optims)
    PATH = f'.net.pth'
    torch.save(net.state_dict(), PATH)
    print(f"Net saved to {PATH}")
    Test(net, data[0][:20], torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), crit)
    # look at outputs after training
    # with torch.no_grad():
    #     # for i, d in enumerate(data[1][0], 0):
    #     d = data[1][50]        
    #     inputs, labels = d[0], d[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #     outputs = net(inputs)
    #     print(f"label: {labels}. output: {outputs}. L1loss: {F.l1_loss(outputs, labels)}. Loss: {crit(outputs, labels)}")
    PATH = 'rankednet.pth'
    torch.save(net.state_dict(), PATH)
    # model.net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    yhat = []
    ypred = []
    for batch in data[0][:4]:
        yhat.extend(batch[1].flatten().tolist())
        ypred.extend(net(batch[0]).flatten().tolist())
    
    print(yhat, ypred)

    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    print(zip(yhat, ypred))
    print("pearsonr:", pearsonr(yhat, ypred))
    print('spearmanr:', spearmanr(yhat, ypred))
    pearson = [
        {'yhat': p1, 'ypred': p} for p1, p in zip(yhat, ypred)
    ]

    df = pd.DataFrame(pearson)

    print(df)
    df.to_csv('resnetcrisprsql200.csv')


'''
- argparse --> specify settings on command line
- put things into if __name__ == '__main__':
- migrate to pytorch lightening
- P0: get this working on a single input!!!
    - track inputs/outputs make sure things match
- 
'''