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
import pytorch_lightning as pl

from dataloaders import dataLoader, Test, NewTrain, accuracy, percentError, rankDataLoader, fullDataLoader
from model import Net, LinearRegression, LogisticRegression, LogisticRegressionClass, NetClass

from to_pl1 import ClassificationModel, Model


if __name__ == '__main__':
    wandb.login()
    targs = ['regular', 'rank']
    print("importing modules thresh")
    pl.seed_everything(42)

    print("compiling data")
    model = ClassificationModel(batch_size=256, mode='target', net=NetClass(1))
    data = model.data
    # import pdb; pdb.set_trace()
    wandb.init(project="crispr-pots", config={
        "optim": "ADAM",
        "loss": "MSE",
        "architecture": f'thresh_target_net',
        "dataset": "CRSIPRSQL",
    })
    trainer = pl.Trainer(auto_scale_batch_size=False, auto_lr_find=True, gpus=(1 if torch.cuda.is_available() else 0))
    # trainer.tune(model)
    wandb.config.update({"Learning Rate": model.learning_rate, "batch_size": model.batch_size})


    # trainer.fit(model)
    # trainer.test(test_dataloaders=data[2])
    # PATH = f'thresh_target_net.pth'
    # torch.save(model.net.state_dict(), PATH)


    for tar in targs:
        print("importing modules", tar)
        pl.seed_everything(42)

        print("compiling data")
        model = Model(batch_size=256, mode='target', target=tar, net=NetClass(1))
        data = model.data
        # import pdb; pdb.set_trace()
        wandb.init(project="crispr-pots", config={
            "optim": "ADAM",
            "loss": "MSE",
            "architecture": f'{mode}_target_net.pth',
            "dataset": "CRSIPRSQL",
        })
        trainer = pl.Trainer(auto_scale_batch_size=False, auto_lr_find=True, gpus=(1 if torch.cuda.is_available() else 0))
        # trainer.tune(model)
        wandb.config.update({"Learning Rate": model.learning_rate, "batch_size": model.batch_size})


        # trainer.fit(model)
        # trainer.test(test_dataloaders=data[2])
        # PATH = f'{mode}_target_net.pth'
        # torch.save(model.net.state_dict(), PATH)
    