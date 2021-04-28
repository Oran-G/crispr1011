

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
from model import Net, LinearRegression, LogisticRegression, basicRNN, NewNet
class Model(pl.LightningModule):
    def __init__(self, batch_size=32, n=1, learning_rate=.01, data=None):
        super(Model, self).__init__()
        # self.net = Net(n)
        # self.net = LogisticRegression()
        # self.net = basicRNN(self.device)

        # used for PL auto batch size and LR scaling
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # print(next(self.net.parameters()).is_cuda)

        # used for calculating total 
        self.trainlabels = None
        self.trainoutputs = None
        self.vallabels = None
        self.valoutputs = None
        self.data = fullDataLoader(batch=self.batch_size)[1] if data == None else data
        self.trainloss = 0
        self.valloss = 0
        self.trainsteps=0
        self.valsteps=0

        self.layer = nn.Linear(20, 40)
        self.l2 = nn.Linear(40, 12)
        self.out = nn.Linear(12, 1)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.log('epoch_val_loss', None)
        self.lstm = nn.LSTM(8, 12, batch_first=False)
        self.do_lstm = True


    # def forward(self, x):

    #     if self.do_lstm == False:
    #         full = torch.cat((x[0], x[1]), dim=3)
    #         # print(full.shape)
    #         hidden = torch.zeros((1, list(full.shape)[0], 12)).type_as(x[0])
    #         # print(hidden.shape)
    #         # print(full[0][:][:][:].shape)
    #         # print(full[:][0][:][:].shape)
    #         # print(full[:][:][0][:].shape)
    #         # print(full[:][:][:][0].shape)
    #         # print('hi', torch.index_select(full, dim=2, index=torch.tensor([0],  dtype=torch.long).to(x[0].device)).shape)
    #         for i in range(list(full.shape)[1]):
    #             # print(full[0][0][i])
    #             hidden = self.l2(self.layer(torch.cat((torch.index_select(full, dim=2, index=torch.tensor([0],  dtype=torch.long).to(x[0].device)), hidden), dim=3)))
    #             # print(hidden)
    #         return torch.sigmoid(self.out(hidden))
    #     else:
    #         full = torch.cat((x[0], x[1]), dim=3).transpose(0, 2)
    #         print(full.shape)
    #         hidden = torch.zeros((list(full.shape)[0], 1,  list(full.shape)[2], 12)).type_as(x[0])
    #         print(hidden.shape)
    #         print(hidden[0].shape)
    #         # print(full.transpose(0, 2).shape)
    #         # torch.index_select(full, dim=2, index=torch.tensor([0],  dtype=torch.long).to(x[0].device)).squeeze(dim=1)
    #         # for i in range(list(full.shape)[1]):
    #         #     print(torch.index_select(full, dim=2, index=torch.tensor([0],  dtype=torch.long).to(x[0].device)).squeeze_(dim=1).shape)
    #         #     print(hidden.shape)
    #         #     out, hidden = self.lstm(torch.index_select(full, dim=2, index=torch.tensor([0],  dtype=torch.long).to(x[0].device)).squeeze(dim=1), hidden)
    #         # return torch.sigmoid(self.out(out))
    #         out, hidden = self.lstm(full.squeeze(dim=1), hidden)
    #         return torch.sigmoid(self.out(out))




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        # inputs, labels = train_batch[0], train_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
        inputs = [
            train_batch['guide'].to(self.device).squeeze_(dim=0), train_batch['target'].to(self.device).squeeze(dim=0)
        ] # (256, 23, 4)
        labels = train_batch['cfd'].to(self.device) # (256, 1)
        outputs = self(inputs) # [254, 1]

        loss = F.mse_loss(outputs, labels)
        
        self.log("train_loss", loss.item())
        self.log("train_accuracy", accuracy(labels, outputs))
        if  self.trainlabels != None:
            torch.cat((self.trainlabels, labels), dim=0)
            try:
                torch.cat((self.trainoutputs, outputs), dim=0)
            except:
                print(self.trainoutputs.shape, outputs.shape)
                quit()
        else:
            self.trainlabels = labels
            self.trainoutputs = outputs
        # result.log('train_loss', loss, prog_bar=True)
        self.trainloss+=loss.item()
        self.trainsteps+=1
        return loss


    # @pl.data_loader 
    def train_dataloader(self):
        return self.data[0]

    def validation_step(self, val_batch, batch_idx):
        # inputs, labels = train_batch[0], train_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
        inputs = [
            val_batch['guide'].to(self.device).squeeze_(dim=0), val_batch['target'].to(self.device).squeeze_(dim=0)
        ] # (256, 23, 4)
        labels = val_batch['cfd'].to(self.device) # (256, 1)
        outputs = self(inputs) # [254, 1]

        loss = F.mse_loss(outputs, labels)
        
        self.log("validation_loss", loss.item())
        self.log("validation_accuracy", accuracy(labels, outputs))
        if self.vallabels != None:
            torch.cat((self.vallabels, labels), dim=0)
            torch.cat((self.valoutputs, outputs), dim=0)
        else:
            self.vallabels = labels
            self.valoutputs = outputs

        self.valloss+=loss.item()
        self.valsteps+=1
        return loss

    # @pl.data_loader
    def val_dataloader(self):
        return self.data[1]

    def on_epoch_end(self):
        print(self.device)
        if self.trainsteps != 0 and self.valsteps != 0:

            self.log('epoch_train_loss', self.trainloss/self.trainsteps)
            self.log('epoch_val_loss', self.valloss/self.valsteps)
            self.log('epoch_train_accuracy', accuracy(self.trainlabels, self.trainoutputs))
            self.log('epoch_validation_accuracy', accuracy(self.vallabels, self.valoutputs))
            self.trainloss = 0
            self.valloss = 0
            self.trainsteps=0
            self.valsteps=0
            self.trainoutputs=None
            self.trainlabels=None
            self.valoutputs=None
            self.vallabels=None











from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    print("importing modules")
    pl.seed_everything(42)
    wandb.login()
    print("compiling data")
    model = Model(batch_size=32)
    wandb_logger = WandbLogger(project="crispr-pots", config={
        "optim": "ADAM",
        "loss": "MSE",
        "architecture": "RNN linear",
        "dataset": "Ranked CRSIPRSQL",
        "Learning Rate": model.learning_rate, 
        "batch_size": model.batch_size,
    })

    
    from pytorch_lightning.callbacks import ModelCheckpoint
    data = model.data
    # import pdb; pdb.set_trace()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='my/path/')
    trainer = pl.Trainer(
        auto_scale_batch_size=True, 
        auto_lr_find=True , 
        gpus=(1 if torch.cuda.is_available() else 0), 
        max_epochs=1250, 
        logger=wandb_logger, 
        # fast_dev_run=1, 
        progress_bar_refresh_rate=5000,
        # callbacks=[checkpoint_callback],
        checkpoint_callback=ModelCheckpoint(
            monitor='epoch_val_loss',
            mode='min',
            save_top_k=1
        ),
        )
    trainer.tune(model)


    trainer.fit(model)
    trainer.test(test_dataloaders=data[0])
    PATH = 'rankednet.pth'
    # torch.save(model.net.state_dict(), PATH)
    # model.net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    yhat = []
    ypred = []
    for i, batch in enumerate(data[0]):
        yhat.extend(batch['cfd'].flatten().tolist())
        ypred.extend(model.net([batch['guide'], batch['target']]).flatten().tolist())
    
    print(yhat, ypred)

    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    # print(zip(yhat, ypred))
    print("pearsonr:", pearsonr(yhat, ypred))
    print('spearmanr:', spearmanr(yhat, ypred))
    pearson = [
        {'yhat': p1, 'ypred': p} for p1, p in zip(yhat, ypred)
    ]

    df = pd.DataFrame(pearson)

    print(df)
    # df.to_csv('resnetcrisprsql100.csv')
    # pd.DataFrame(zip(yhat, ypred), columns=['yhat', 'ypred'])




'''

TODO: at the end of each epoch:
+ save the model checkpoint
+ pearson / spearman correlation
+ write a dataframe of the results
    --> def compute_dataframe(df: pd.DataFrame, checkpoint_path: str):
            model = load_model(checkpoint_path)
            guides, targets = df.guide_seqs.tolist(), df.targets.tolist()
            preds = []
            for guide, target in zip(guides, targets):
                pred = model([guide, target])
                preds.append(pred)
            df['pred'] = preds
            return df
    
    init_df = pd.from_csv('...crispr_sql.csv')
    new_df = compute_dataframe(init_df, '...model.pt')
    new_df.to_csv('.crispr_sql.processed.csv')


test/hps/epoch/point/pred

test/hps/4/354/0.08

https://pastebin.pl/view/7512e167

'''