

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
from model import Net, LinearRegression, LogisticRegression
class Model(pl.LightningModule):
    def __init__(self, batch_size=32, n=1, learning_rate=.02, data=None):
        super(Model, self).__init__()
        # self.net = Net(n)
        self.net = LogisticRegression()
        if torch.cuda.is_available():   
            self.net.cuda()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # print(next(self.net.parameters()).is_cuda)
        self.trainlabels = []
        self.trainoutputs = []
        self.vallabels = []
        self.valoutputs = []
        self.data = fullDataLoader(batch=self.batch_size)[1] if data == None else data
        self.optim = self.configure_optimizers()
    def forward(self, x):
        # print(x[0].size())
        embedding = self.net(x)
        return embedding
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        # inputs, labels = train_batch[0], train_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
        inputs = [
            train_batch['guide'], train_batch['target']
        ] # (256, 23, 4)
        labels = train_batch['cfd'] # (256, 1)
        # print(inputs[0].shape)
        # self.optim.zero_grad()
        # inputs[0] = torch.squeeze(inputs[0], 1)
        # inputs[1] = torch.squeeze(inputs[1], 1)
        # labels = torch.squeeze(labels, 1)
        outputs = self.net(inputs) # [254, 1]
        # import pdb; pdb.set_trace()

        loss = F.mse_loss(outputs, labels)# / labels.size(0)
        # loss.backward()
        # self.optim.step()
        # w = {f'output {i+1}': outputs.flatten()[i] for i in range(outputs.flatten().size(0))}
        # w.update({
        #     f'label {i+1}': labels.flatten()[i] for i in range(labels.flatten().size(0))
        #     })
        w = ({
            'train_loss': loss.item(), 
            'accuracy': accuracy(labels, outputs), 
            # 'percent error': percentError(outputs, labels),
        })
        # self.log("train_loss", loss)
        wandb.log(w)
        # print(train_batch[0][0].size())
        return loss#, {
        #     'stats': {
        #         'loss': loss,
        #         'acc': acc,
        #         'pearsonr', pearsonr(labels, outputs)
        #         ...

        #     },
        #     'data': {
        #         'inputs': inputs,
        #         'labels': labels,
        #     }
        # }

    # def train_end(self, outputs):
    #     def cat_data(key):
    #         return torch.stack([output['data'][key] for output in outputs])

    #     return {
    #         'stats': {
    #             'pearsonr': pearsonr(cat_data('inputs'), cat_data('labels'))
    #         }
    #     }


    # @pl.data_loader 
    def train_dataloader(self):
        return self.data[0]

    # def validation_step(self, val_batch, batch_idx):
    #     # inputs, labels = val_batch[0], val_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
    #     inputs = [
    #         val_batch['guide'], val_batch['target']
    #     ]
    #     labels = val_batch['cfd']



    #     # print(type(inputs[0]))
    #     # inputs[0] = torch.squeeze(inputs[0], 1)
    #     # inputs[1] = torch.squeeze(inputs[1], 1)
    #     # labels = torch.squeeze(labels, 1)
    #     # print(inputs[0].size())
    #     # print(type(self.net.parameters()))
    #     outputs = self.net(inputs)
    #     loss = F.mse_loss(outputs, labels)# / labels.size(0)
    #     # w = {f'output {i+1}': outputs.flatten()[i] for i in range(outputs.flatten().size(0))}
    #     # w.update({
    #     #     f'label {i+1}': labels.flatten()[i] for i in range(labels.flatten().size(0))
    #     #     })
    #     w = ({'val_loss': loss.item(), 
    #         'accuracy': accuracy(labels, outputs), 
    #         # 'percent error': percentError(outputs, labels)
    #     })
    #     # self.log("val_loss", loss)
    #     wandb.log(w)
    #     return loss

    # # @pl.data_loader
    # def val_dataloader(self):
    #     return self.data[1]













if __name__ == '__main__':
    print("importing modules logistic")
    pl.seed_everything(42)
    wandb.login()

    print("compiling data")
    model = Model(batch_size=25)
    data = model.data
    # import pdb; pdb.set_trace()
    wandb.init(project="crispr-pots", config={
        "optim": "ADAM",
        "loss": "MSE",
        "architecture": "RESNET 1",
        "dataset": "Ranked CRSIPRSQL :1",
    })
    trainer = pl.Trainer(auto_scale_batch_size=True, auto_lr_find=False , gpus=(1 if torch.cuda.is_available() else 0), max_epochs=20)
    # trainer.tune(model)
    wandb.config.update({"Learning Rate": model.learning_rate, "batch_size": model.batch_size})


    trainer.fit(model)
    # trainer.test(test_dataloaders=data[0])
    PATH = 'rankedlogistic.pth'
    torch.save(model.net.state_dict(), PATH)
    # model.net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    yhat = []
    ypred = []
    for i, batch in enumerate(data[0]):
        yhat.extend(batch['cfd'].flatten().tolist())
        ypred.extend(model.net([batch['guide'], batch['target']]).flatten().tolist())
    
    # print(yhat, ypred)

    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    # print(zip(yhat, ypred))
    # print("pearsonr:", pearsonr(yhat, ypred))
    # print('spearmanr:', spearmanr(yhat, ypred))
    pearson = [
        {'yhat': p1, 'ypred': p} for p1, p in zip(yhat, ypred)
    ]

    df = pd.DataFrame(pearson)

    print(df)
    df.to_csv('logistic.csv')
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

'''





class ClassificationModel(pl.LightningModule):
    def __init__(self, batch_size=32, n=2, learning_rate=.02, data=None):
        super(Model, self).__init__()
        # self.net = Net(n)
        self.net = LogisticRegression()
        if torch.cuda.is_available():   
            self.net.cuda()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # print(next(self.net.parameters()).is_cuda)
        self.trainlabels = []
        self.trainoutputs = []
        self.vallabels = []
        self.valoutputs = []
        self.data = fullDataLoader(batch=self.batch_size)[1] if data == None else data
    
    def forward(self, x):
        # print(x[0].size())
        embedding = self.net(x)
        return embedding
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        # inputs, labels = train_batch[0], train_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
        inputs = [
            train_batch['guide'], train_batch['target']
        ] # (256, 23, 4)
        labels = train_batch['cfd'] # (256, 1)
        # print(inputs[0].shape)

        # inputs[0] = torch.squeeze(inputs[0], 1)
        # inputs[1] = torch.squeeze(inputs[1], 1)
        # labels = torch.squeeze(labels, 1)
        outputs = self.net(inputs) # [254, 1]
        # import pdb; pdb.set_trace()

        loss = F.cross_entropy(outputs, labels)# / labels.size(0)
        # w = {f'output {i+1}': outputs.flatten()[i] for i in range(outputs.flatten().size(0))}
        # w.update({
        #     f'label {i+1}': labels.flatten()[i] for i in range(labels.flatten().size(0))
        #     })
        w = ({
            'train_loss': loss.item(), 
            'accuracy': (predicted == labels).sum().item() / labels.size[0], 
            # 'percent error': percentError(outputs, labels),
        })
        # self.log("train_loss", loss)
        wandb.log(w)
        # print(train_batch[0][0].size())
        return loss#, {
        #     'stats': {
        #         'loss': loss,
        #         'acc': acc,
        #         'pearsonr', pearsonr(labels, outputs)
        #         ...

        #     },
        #     'data': {
        #         'inputs': inputs,
        #         'labels': labels,
        #     }
        # }

    # def train_end(self, outputs):
    #     def cat_data(key):
    #         return torch.stack([output['data'][key] for output in outputs])

    #     return {
    #         'stats': {
    #             'pearsonr': pearsonr(cat_data('inputs'), cat_data('labels'))
    #         }
    #     }


    # @pl.data_loader 
    def train_dataloader(self):
        return self.data[0]

    def validation_step(self, val_batch, batch_idx):
        # inputs, labels = val_batch[0], val_batch[1]#.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )
        inputs = [
            val_batch['guide'], val_batch['target']
        ]
        labels = val_batch['cfd']



        # print(type(inputs[0]))
        # inputs[0] = torch.squeeze(inputs[0], 1)
        # inputs[1] = torch.squeeze(inputs[1], 1)
        # labels = torch.squeeze(labels, 1)
        # print(inputs[0].size())
        # print(type(self.net.parameters()))
        outputs = self.net(inputs)
        loss = F.mse_loss(outputs, labels)# / labels.size(0)
        # w = {f'output {i+1}': outputs.flatten()[i] for i in range(outputs.flatten().size(0))}
        # w.update({
        #     f'label {i+1}': labels.flatten()[i] for i in range(labels.flatten().size(0))
        #     })
        w = ({'val_loss': loss.item(), 
            'accuracy': (predicted == labels).sum().item() / labels.size[0], 
            # 'percent error': percentError(outputs, labels)
        })
        # self.log("val_loss", loss)
        wandb.log(w)
        return loss

    # @pl.data_loader
    def val_dataloader(self):
        return self.data[1]