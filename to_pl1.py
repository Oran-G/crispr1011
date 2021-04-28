

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
from model import Net, LinearRegression, LogisticRegression, LogisticRegressionClass


class ClassificationModel(pl.LightningModule):
    def __init__(self, batch_size=32, n=2, learning_rate=.02, data=None, mode='target', net=None):
        super(ClassificationModel, self).__init__()
        # self.net = Net(n)
        if net == None:
            self.net = LogisticRegressionClass()
        else:
            self.net=net
        if torch.cuda.is_available():   
            self.net.cuda()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # print(next(self.net.parameters()).is_cuda)
        self.trainlabels = []
        self.trainoutputs = []
        self.vallabels = []
        self.valoutputs = []
        self.loss = nn.CrossEntropyLoss()
        self.data = fullDataLoader(batch=self.batch_size, mode=mode)[1] if data == None else data
    
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
        print(labels.shape, outputs.shape)

        loss = self.loss(outputs, labels)# / labels.size(0)
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
        print(labels.shape, outputs.shape)
        loss = self.loss(outputs, labels)# / labels.size(0)
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


class Model(pl.LightningModule):
    def __init__(self, batch_size=32, n=2, learning_rate=.02, data=None, mode='target', target='rank', net=None):
        super(Model, self).__init__()
        
        if net == None:
            self.net = LogisticRegression()
        else:
            self.net=net
        if torch.cuda.is_available():   
            self.net.cuda()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # print(next(self.net.parameters()).is_cuda)
        self.trainlabels = []
        self.trainoutputs = []
        self.vallabels = []
        self.valoutputs = []
        self.data = fullDataLoader(batch=self.batch_size, mode=mode, target=target)[1] if data == None else data
    
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

        loss = F.mse_loss(outputs, labels)# / labels.size(0)
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
        return loss

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
            'accuracy': accuracy(labels, outputs), 
            # 'percent error': percentError(outputs, labels)
        })
        # self.log("val_loss", loss)
        wandb.log(w)
        return loss

    # @pl.data_loader
    def val_dataloader(self):
        return self.data[1]













if __name__ == '__main__':
    wandb.login()
    targs = ['regular', 'rank']
    modes = ['target', 'study', 'guide']
    for mode in modes:
        print("importing modules")
        pl.seed_everything(42)

        print("compiling data")
        model = ClassificationModel(batch_size=256, mode=mode)
        data = model.data
        # import pdb; pdb.set_trace()
        wandb.init(project="crispr-pots", config={
            "optim": "ADAM",
            "loss": "MSE",
            "architecture": f'{mode}_thresh_logistic',
            "dataset": "CRSIPRSQL",
        })
        trainer = pl.Trainer(auto_scale_batch_size=False, auto_lr_find=True, gpus=(1 if torch.cuda.is_available() else 0))
        trainer.tune(model)
        wandb.config.update({"Learning Rate": model.learning_rate, "batch_size": model.batch_size})


        trainer.fit(model)
        trainer.test(test_dataloaders=data[2])
        PATH = f'{mode}_thresh_logistic.pth'
        torch.save(model.net.state_dict(), PATH)


    for tar in targs:
        for mode in modes:
            print("importing modules")
            pl.seed_everything(42)

            print("compiling data")
            model = Model(batch_size=256, mode=mode, target=tar)
            data = model.data
            # import pdb; pdb.set_trace()
            wandb.init(project="crispr-pots", config={
                "optim": "ADAM",
                "loss": "MSE",
                "architecture": f'{mode}_{tar}_logistic.pth',
                "dataset": "CRSIPRSQL",
            })
            trainer = pl.Trainer(auto_scale_batch_size=False, auto_lr_find=True, gpus=(1 if torch.cuda.is_available() else 0))
            trainer.tune(model)
            wandb.config.update({"Learning Rate": model.learning_rate, "batch_size": model.batch_size})


            trainer.fit(model)
            trainer.test(test_dataloaders=data[2])
            PATH = f'{mode}_{tar}_logistic.pth'
            torch.save(model.net.state_dict(), PATH)
    


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



