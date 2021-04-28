import random
import copy
random.seed(42)
import csv
import torch
import time
import statistics
import wandb
from model import Net, LinearRegression, LogisticRegression

def byGuide(data, val=None, test=None):
    val_guides = val
    if val == None:
        val_guides = [
        "GGGTGGGGGGAGTTTGCTCCTGG",
        "GACCCCCTCCACCCCGCCTCCGG",
        "GGCCTCCCCAAAGCCTGGCCAGG",
        "GAACACAAAGCATAGACTGCGGG"
        
        ]
    test_guides = test
    if test==None:
        test_guides = [
            "GCAAAACTCAACCCTACCCCAGG",
            "GGCCCAGACTGAGCACGTGATGG",
            "GGGAAAGACCCAGCATCCGTGGG",
            "GGAATCCCTTCTGCAGCACCTGG",
            "GTGAGTGAGTGTGTGCGTGTGGG",
            "GATGATGATGCCCCGGGCGTTGG",
            "GCCGGAGGGGTTTGCACAGAAGG"
        ]
    
    train_set =  []
    val_set = []
    test_set = []
    for pair in data:
        pair['off'] = torch.tensor([1., 0.])
        if pair['grna_target_sequence'] in val_guides:
            val_set.append(pair)
        elif pair['grna_target_sequence'] in test_guides:
            test_set.append(pair)
        else:   
            train_set.append(pair)
    return [train_set, val_set, test_set]        

def byTarget(data, train=.7, val=.1, test=.2):
    random.shuffle(data)
    train_set =  []
    val_set = []
    test_set = []
    for i in range(len(data)):
        if i <= len(data) * train:
            train_set.append(data[i])
        elif i <= len(data) * (train + val):
            val_set.append(data[i])
        else:
            test_set.append(data[i])
    return [train_set, val_set, test_set]           




def byStudy(data, val=None, test=None):
    val_studies = val
    if val == None:
        val_studies = [
            'Anderson',
            'Ran',
            
        ]
    test_studies = test
    if test==None:
        test_studies = [
            'Kim',
            'Tsai',
            'Cho',
        ]
    train_set =  []
    val_set = []
    test_set = []
    for pair in data:
        pair['off'] = torch.tensor([1., 0.])
        if pair['study_name'] in val_studies:
            val_set.append(pair)
        elif pair['study_name'] in test_studies:
            test_set.append(pair)
        else:   
            train_set.append(pair)
    return [train_set, val_set, test_set]  



def one_hot(data, sign='+'):
    sins = None
    sequence = None
    data = data.lower()
    for n in data:
        
        one_hot = torch.zeros((1, 4))
        if n =='a':
            one_hot[0][0] = 1
        elif n == 'c':
            one_hot[0][1] = 1
        elif n == 'g':
            one_hot[0][2] = 1
        elif n == 't':
            one_hot[0][3] = 1
        if sins == None:
            sequence = copy.deepcopy(one_hot)
            sins = 1
        else:
            sequence = torch.cat((sequence, one_hot), dim=0)
    if list(sequence.size())[0] < 23:
        for i in range(23 - list(sequence.size())[0]):
            sequence = torch.cat((sequence, torch.zeros((1, 4))), dim=0)  
    if list(sequence.size())[0] > 23: 
        sequence = sequence[:23]
    if sign == '-':
        sequence = torch.flip(sequence, [1])      
    return sequence   

        
# import numpy as np

def dataLoader(file="crisprsql.csv", batch=64, mode="target"):
    ftime = time.monotonic()
    with open(file) as f:
        d = list(csv.DictReader(f))
        if mode == "study":
            loadData = byStudy(d)
        elif mode == "guide":
            loadData = byGuide(d)
        else:
            loadData = byTarget(d)
    data = list()
    dl = list()
    train = True
    for t in range(3):
        average_value = list()
        thisdata = list()
        for line in loadData[t]:
            if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
                thisdata.append([
                    [one_hot(line['grna_target_sequence'], line['grna_target_strand']), 
                        one_hot(line['target_sequence'], line["target_strand"])],
                    torch.tensor([float(line['cleavage_freq'])])])
                average_value.append(float(line['cleavage_freq']))   
            # if line 


        # mode = 0
        # zero = 0
        # for p in average_value:
        #     if p == statistics.mode(average_value):
        #         mode+=1
        #     if p <0:
        #         zero+=1 
        # print(f"average CFD of {len(average_value)} datapoints in set {t + 1}: {sum(average_value)/len(average_value)}.\nMedian: {statistics.median(average_value)}.\nMode: {statistics.mode(average_value)} with {mode} datapoint.\nstandard deviation: {statistics.pstdev(average_value)}.\nlowest value: {min(average_value)}.\nHighest value: {max(average_value)}\n{zero} datapoints below zero\n\n")
        if train == True:
            dl.append(torch.utils.data.DataLoader(thisdata, batch, True, num_workers=(4 if torch.cuda.is_available() else 4)))
            print(thisdata[0][0][0].size())
            train = False
        else:
            dl.append(torch.utils.data.DataLoader(thisdata, batch, False, num_workers=(4 if torch.cuda.is_available() else 4)))
        
        thisdata1 = list()            
        for i in range(int(len(thisdata)/batch)):
            ones = None
            twos = None
            threes = None
            for j in range(batch):
                
                if  ones == None:
                    ones = thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)
                    twos = thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)
                    threes = thisdata[(i * batch) + j][1].unsqueeze_(0)
                else:
                    ones = torch.cat((ones, thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)), dim=0)  
                    twos = torch.cat((twos, thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)), dim=0) 
                    threes = torch.cat((threes, thisdata[(i * batch) + j][1].unsqueeze_(0)), dim=0)      
                       
            thisdata1.append([[ones, twos], threes])          


        data.append(thisdata1) 
 
    print('time to load data: ', time.monotonic() - ftime, 'seconds')   
    

    return [data, dl]

# from scipy.stats import rankdata

class CRISPRDataset(torch.utils.data.Dataset):
    def __init__(self, thisdata):
        self.thisdata = thisdata
    
    def __len__(self):
        return len(self.thisdata)

    def __getitem__(self, idx):
        item = self.thisdata[idx]
        sample = {
            # (23, 4)
            'target': torch.squeeze(item[0][1]).unsqueeze_(dim=0),
            'guide': torch.squeeze(item[0][0]).unsqueeze_(dim=0),
            # (1)
            'cfd': torch.squeeze(item[1]).unsqueeze_(dim=0)
        }
        return sample

    
def collate_fn(batch):
    # (256, 23, 4)
    # (256, 1)
    # print(sum(list(batch[0]['cfd'].shape)), sum(list(batch[0]['target'].shape, sum(list(batch[0]['guide'].shape)))))

    output = {}

    b = {key: [] for key in batch[0].keys()}
    for i in batch:
        if sum(list(i['cfd'].shape)) > 0 and sum(list(i['target'].shape)) > 0  and sum(list(i['guide'].shape)) > 0 :
            for key in i.keys():
                b[key].append(i[key])
        else:
            print('1', sum(list(i['cfd'].shape)), i['cfd'])
            print('2', sum(list(i['target'].shape)),  len(i['target'].shape), i['target'].tolist())
            print('3', sum(list(i['guide'].shape)),  len(i['guide'].shape))

    for key in b.keys():
        # print(b[key])s
        if len(b[key]) > 0:
            output[key] = torch.stack(b[key])
        else:
            output[key] = torch.tensor([])






    # output = {
    #     key: torch.stack([batch[i][key] for i in range(len(batch)) \
    #             if all( len(batch[i][k].shape) > 0 for k in batch[0].keys() )
    #     ])
    #     for key in batch[0].keys()
    # }

    return output
import pandas as pd

def rankDataLoader(file="crisprsql.csv", batch=64, mode="target"):
    ftime = time.monotonic()
    with open(file) as f:
        d = list(csv.DictReader(f))
        if mode == "study":
            loadData = byStudy(d)
        elif mode == "guide":
            loadData = byGuide(d)
        else:
            loadData = byTarget(d)
    data = list()
    dl = list()
    train = True
    ranks = list()
    for line in d:
        if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
            ranks.append(float(line['cleavage_freq']))
    ranks.sort()
    for t in range(3):
        
        df = pd.DataFrame(loadData[t])

        # df.drop(df.columns.difference(['cleavage_freq']), 1, inplace=True)
        # pd.to_numeric(df['cleavage_freq']
        pd.to_numeric(df.cleavage_freq,  errors='coerce')
        # cleave = df.cleavage_freq
        
        # df_ = pd.DataFrame(loadData[t]).drop(['cleavage_freq'], 1, inplace=True)
        # df_.join(cleave)
        df.dropna(subset=['cleavage_freq'], inplace=True)
        print(df.head())
        average_value = list()
        thisdata = list()
        for line in df.to_dict("records"):
            if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
                thisdata.append([
                    [one_hot(line['grna_target_sequence'], line['grna_target_strand']), 
                        one_hot(line['target_sequence'], line["target_strand"])],
                    torch.tensor(ranks.index(float(line['cleavage_freq'])) / len(ranks))])
                average_value.append(float(line['cleavage_freq']))   
            # if line 


        # mode = 0
        # zero = 0
        # for p in average_value:
        #     if p == statistics.mode(average_value):
        #         mode+=1
        #     if p <0:
        #         zero+=1 
        # print(f"average CFD of {len(average_value)} datapoints in set {t + 1}: {sum(average_value)/len(average_value)}.\nMedian: {statistics.median(average_value)}.\nMode: {statistics.mode(average_value)} with {mode} datapoint.\nstandard deviation: {statistics.pstdev(average_value)}.\nlowest value: {min(average_value)}.\nHighest value: {max(average_value)}\n{zero} datapoints below zero\n\n")
        if train == True:
            # dl.append(torch.utils.data.DataLoader(thisdata, batch, True, num_workers=(1 if torch.cuda.is_available() else 0)))
            dl.append(torch.utils.data.DataLoader(CRISPRDataset(thisdata), batch, True, collate_fn=collate_fn, num_workers=(1 if torch.cuda.is_available() else 0)))
            
            # print(thisdata[0][0][0])
            train = False
        else:
            # dl.append(torch.utils.data.DataLoader(thisdata, batch, False, num_workers=(1 if torch.cuda.is_available() else 0)))
            dl.append(torch.utils.data.DataLoader(CRISPRDataset(thisdata), batch, False, collate_fn=collate_fn, num_workers=(1 if torch.cuda.is_available() else 0)))
        # import pdb; pdb.set_trace()
        thisdata1 = list()            
        for i in range(int(len(thisdata)/batch)):
            ones = None
            twos = None
            threes = None
            for j in range(batch):
                
                if  ones == None:
                    ones = thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)
                    twos = thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)
                    threes = thisdata[(i * batch) + j][1].unsqueeze_(0)
                else:
                    ones = torch.cat((ones, thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)), dim=0)  
                    twos = torch.cat((twos, thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)), dim=0) 
                    threes = torch.cat((threes, thisdata[(i * batch) + j][1].unsqueeze_(0)), dim=0)      
                       
            thisdata1.append([[ones, twos], threes])          


        data.append(thisdata1) 
 
    print('time to load data: ', time.monotonic() - ftime, 'seconds')   
    
    return [data, dl]







def fullDataLoader(file="augmentcrisprsql.csv", batch=64, mode="target", target='rank'):
    ftime = time.monotonic()
    with open(file) as f:
        d = list(csv.DictReader(f))
        random.shuffle(d)
        if mode == "study":
            loadData = byStudy(d)
        elif mode == "guide":
            loadData = byGuide(d)
        else:
            loadData = byTarget(d)
    data = list()
    dl = list()
    train = True
    for t in range(3):
        
        average_value = list()
        thisdata = list()
        q = 0
        for line in loadData[t]:
            if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:

                if target == 'regular':
                    label = float(line['cleavage_freq'])
                elif target == 'rank':
                    label = [float(line['ranked_cleavage_freq'])]
                else:
                    label = [0, 1] if float(line['threshhold_cleavage_freq']) == 0 else [1, 0]

                if sum(list(torch.tensor([label]).shape)) > 0 and sum(list(one_hot(line['grna_target_sequence'], line['grna_target_strand']).shape)) > 0  and sum(list(one_hot(line['target_sequence'], line["target_strand"]).shape)) > 0:
                    thisdata.append([
                        [one_hot(line['grna_target_sequence'], line['grna_target_strand']), 
                            one_hot(line['target_sequence'], line["target_strand"])],
                        torch.tensor(label)])
                    average_value.append(label)
                    # print(sum(list(torch.tensor([label]).shape)), sum(list(one_hot(line['grna_target_sequence'], line['grna_target_strand']).shape)), sum(list(one_hot(line['target_sequence'], line["target_strand"]).shape)))
                    
                else:
                    q+=1
                    print(sum(list(torch.tensor([label]).shape)), sum(list(one_hot(line['grna_target_sequence'], line['grna_target_strand']).shape)), sum(list(one_hot(line['target_sequence'], line["target_strand"]).shape)))
                    # print(torch.tensor([label), len(torch.tensor([label]).shape))
        print(q)
            # if line 


        # mode = 0
        # zero = 0
        # for p in average_value:
        #     if p == statistics.mode(average_value):
        #         mode+=1
        #     if p <0:
        #         zero+=1 
        # print(f"average CFD of {len(average_value)} datapoints in set {t + 1}: {sum(average_value)/len(average_value)}.\nMedian: {statistics.median(average_value)}.\nMode: {statistics.mode(average_value)} with {mode} datapoint.\nstandard deviation: {statistics.pstdev(average_value)}.\nlowest value: {min(average_value)}.\nHighest value: {max(average_value)}\n{zero} datapoints below zero\n\n")
        if train == True:
            # dl.append(torch.utils.data.DataLoader(thisdata, batch, True, num_workers=(1 if torch.cuda.is_available() else 0)))
            dl.append(torch.utils.data.DataLoader(CRISPRDataset(thisdata), batch, True, collate_fn=collate_fn, num_workers=4))
        
            # print(thisdata[0][0][0])
            train = False
        else:
            # dl.append(torch.utils.data.DataLoader(thisdata, batch, False, num_workers=(1 if torch.cuda.is_available() else 0)))
            dl.append(torch.utils.data.DataLoader(CRISPRDataset(thisdata), batch, False, collate_fn=collate_fn, num_workers=4))
        # import pdb; pdb.set_trace()
        thisdata1 = list()            
        for i in range(int(len(thisdata)/batch)):
            ones = None
            twos = None
            threes = None
            for j in range(batch):
                
                if  ones == None:
                    ones = thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)
                    twos = thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)
                    threes = thisdata[(i * batch) + j][1].unsqueeze_(0)
                else:
                    ones = torch.cat((ones, thisdata[(i * batch) + j][0][0].unsqueeze_(0).unsqueeze_(0)), dim=0)  
                    twos = torch.cat((twos, thisdata[(i * batch) + j][0][1].unsqueeze_(0).unsqueeze_(0)), dim=0) 
                    threes = torch.cat((threes, thisdata[(i * batch) + j][1].unsqueeze_(0)), dim=0)      
                        
            thisdata1.append([[ones, twos], threes])          


        data.append(thisdata1) 

    print('time to load data: ', time.monotonic() - ftime, 'seconds')   

    return [data, dl]








from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
def roc(labels, outputs):
    llabels = labels.flatten().tolist()
    loutputs = outputs.flatten().tolist()
    average_values = dict()
    # print(len(llabels), len(loutputs))
    for i in range(1, 2):
        thislabel = list()
        thisoutput = list()
        pres = 0
        totalpres = 0
        for j in range(len(llabels)):

            if llabels[j] <= .01 / i:
                thislabel.append(0)
            else:
                thislabel.append(1)    
            if loutputs[j] <= .01 / i:
                thisoutput.append(0)
            else:
                thisoutput.append(1)
            if thislabel[-1] == thisoutput[-1]:
                pres += 1
            totalpres +=1        
        lr_precision, lr_recall, _ = precision_recall_curve(thislabel, thisoutput)
        average_values[.1/i] = [roc_auc_score(thislabel, thisoutput), auc(lr_recall, lr_precision), pres/totalpres]
    return average_values    


def accuracy(labels, outputs, percent=.10):
    llabels = labels.flatten().tolist()
    loutputs = outputs.flatten().tolist()
    correct = 0
    total = 0
    # print(llabels)
    for i in range(len(llabels)):
        if llabels[i] * (1 - percent) <= loutputs[i] and llabels[i] * (1 + percent) >= loutputs[i]:
            correct +=1
        total += 1

    return correct / total    


def percentError(outputs, labels):
    return torch.mean(torch.abs(labels - outputs) / labels)


                


                


def Test(net, dataset, device, crit, logpath=None):
    
    net.eval()
    correct = 0
    total = 0
    totalloss = 0
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):
            inputs, labels = data[0], data[1].to(device) 
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            totalloss+=1
            correct += (predicted == labels).sum().item()
            loss+=crit(outputs, labels)
    if logpath!= None:
        f = open(logpath, 'w')
        f.write('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        f.write(f"total: {total} correct: {correct}")
        f.write(f'loss: {loss/totalloss}')
        f.close()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print(f"total: {total} correct: {correct}") 
    print(f'loss: {loss/totalloss}')
    return 100 * correct / total        

def getAllStudy():
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()
        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"]:
                try:
                    alls[row['study_name']].add(row['grna_target_sequence'])   
                except KeyError:
                    alls[row["study_name"]] = set(row['grna_target_sequence'])    
        for r in alls:
            print(r)
            print(alls[r])
            print(len(alls[r]))
        

def getallGuide():
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()

        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"]:
                try:
                    alls[row['grna_target_sequence']].add(row['target_sequence'])   
                except KeyError:
                    alls[row["grna_target_sequence"]] = set(row['target_sequence'])    
        for r in alls:
            print(r)
            print(alls[r])
            print(len(alls[r]))
        

def aboveandbelow(threshold):
    with open("crisprsql.csv") as f:
        data = csv.DictReader(f)
        alls = dict()
        above = 0
        total = 0
        for row in data:
            if row['grna_target_sequence'] not in ["C", 'G', 'A', "T"] and row['cleavage_freq'] != '':
                if float(row['cleavage_freq']) > threshold:
                    above+=1
                total+=1
    

    print(f'Above: {above / total}%. Below: {(total - above) / total}')







def NewTrain(epochs, optim, crit, batch_per, train_data, val_data, net, device, optim_time=None, logpath=None):
    net.to(device)
    #def optim, loss, and init graph data
    criterion = crit
    optimizer = optim
    # get all labels for ROC
    full_full_labels = None
    for i, data in enumerate(train_data, 0):
        if full_full_labels == None:
            full_full_labels = data[1].to(device) 
        else:
            full_full_labels = torch.cat((full_full_labels, data[1].to(device)), 0)   
    full_val_labels = None         
    for i, data in enumerate(val_data, 0):
        if full_val_labels == None:
            full_val_labels = data[1].to(device) 
        else:
            full_val_labels = torch.cat((full_val_labels, data[1].to(device)), 0)            
    print("begin training")
    if logpath!= None:
        f = open(logpath, 'w')
    #these go down, and random loss is ~2.303 so 15 will be replaced
    best = 15
    bestval = 15
    bestepoch = 0
    e = 0
    # begin training loop, larget loop is for lr scedule
    times = list()
    # bestnet = LogisticRegression()
    # bestnet.load_state_dict(copy.deepcopy(net.state_dict()))
    for q in optim_time:
        optimizer = q[1]
        print(q[0])
        # net.load_state_dict(copy.deepcopy(bestnet.state_dict())
        # print(
        #     'params', [p for p in net.parameters()], 
        #     '\ngrads', [p.grad for p in net.parameters()] 
        # )
        # epoch loop
        for epoch in range(q[0]):  # loop over the dataset multiple times
            ftime = time.monotonic()
            random.shuffle(train_data)
            correct = 0
            total = 0
            running_loss = 0.0
            # train mode
            net.train()
            full_output = None
            full_labels = None
            full_full_output = None
            
            for i, data in enumerate(train_data, 0):
                
                # train step
                inputs, labels = data[0], data[1].to(device) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # t = time.monotonic()
                outputs = net(inputs)
                # print(time.monotonic - t, " seconds for 512 outputs")
                loss = criterion(outputs, labels)
                loss.backward()
                # import pdb; pdb.set_trace()
                # things to look at:
                # - loss
                # - parameters
                # - inputs
                # - grads
                # if e % 300 == 299:

                # print(
                #     'loss', loss, 
                #     # '\ninputs', inputs,
                #     '\nlabels', labels,
                #     '\noutputs', outputs
                # )
                    
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)   
                correct += (predicted == labels).sum().item()
                # print()
                
                running_loss += loss.item()
                if full_output == None:
                    full_output = outputs
                else:
                    full_output = torch.cat((full_output, outputs), 0)

                if full_labels == None:
                    full_labels = labels
                else:
                    full_labels = torch.cat((full_labels, labels), 0)  
                # w = {f'output {i}': outputs.flatten()[i] for i in range(outputs.flatten().size(0))}
                # w.update({
                #     f'label {i}': labels.flatten()[i] for i in range(labels.flatten().size(0))
                # })
                w = ({'loss': loss.item(), 
                    'accuracy': accuracy(labels, outputs),
                    'percent error': percentError(outputs, labels)})
                wandb.log(
                    # {
                    #     'loss': loss.item(), 
                    #     # 'params': [p for p in net.parameters()], 
                    #     # 'grads': [p.grad for p in net.parameters()], 
                    #     # 'inputs': inputs,
                    #     f'label {i}': labels.flatten()[i] for i in len(labels.flatten().size(0)),
                    #     f'output {i}': outputs.flatten()[i] for i in len(outputs.flatten().size(0)),
                    #     'accuracy': accuracy(labels, outputs)
                    # }
                    w
                )
                # print statistics
                if i % batch_per == batch_per - 1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (e + 1, i + 1, running_loss / batch_per))
                    # best = min(best, running_loss / batch_per)
                    
                    # print('Accuracy of the network on the ' + str(batch_per) + 'th update: %d %%' % (
                    #     100 * correct / total))
                    
                    wl = roc(full_labels, full_output)
                    wandlog = {}
                    for q in wl:
                        wandlog[f"midepoch ROC_AUC"] = wl[q][0]
                        wandlog[f"midepoch PR_AUC"] = wl[q][1]
                        wandlog[f"midepoch threshhold accuracy"] = wl[q][2]



                    # wandlog.update({
                    #     "LOSS": running_loss / batch_per, 
                    #     "TYPE": "TRAIN", 
                    #     'EPOCH': e+1, 
                    #     'UPDATE': (e*len(train_data)) + i + 1})
                    w.update({'midepoch loss': loss.item(), 
                        'midepoch accuracy': accuracy(labels, outputs),
                        'midepoch percent error': percentError(outputs, labels)})
                    wandb.log(
                        # {
                        #     'loss': loss.item(), 
                        #     # 'params': [p for p in net.parameters()], 
                        #     # 'grads': [p.grad for p in net.parameters()], 
                        #     # 'inputs': inputs,
                        #     f'label {i}': labels.flatten()[i] for i in len(labels.flatten().size(0)),
                        #     f'output {i}': outputs.flatten()[i] for i in len(outputs.flatten().size(0)),
                        #     'accuracy': accuracy(labels, outputs)
                        # }
                        w
                    )
                    wandb.log(wandlog)
                    if full_full_output == None:
                        full_full_output = full_output
                    else:
                        full_full_output = torch.cat((full_full_output, full_output), 0)  
                    
                    full_output = None
                    full_labels = None


                    running_loss = 0
                    correct = 0
                    total = 0
            # print('[%d] loss: %.20f' %
            # (epoch + 1, running_loss / total))
            # if logpath != None:
            #     f.write('[%d] loss: %.20f' %
            # (epoch + 1, running_loss / total))   
            if full_full_output == None:
                full_full_output = full_output
            else:
                full_full_output = torch.cat((full_full_output, full_output), 0)  
            # ROC is commented out when training on 10 samples
            wl = roc(full_full_labels, full_full_output)
            w = {}

            for q in wl:
                w[f"epoch ROC_AUC"] = wl[q][0]
                w[f"epoch PR_AUC"] = wl[q][1]
                w[f"epoch threshhold accuracy"] = wl[q][2]
            # wandlog.update({
            #     "LOSS": running_loss / batch_per, 
            #     "TYPE": "TRAIN", 
            #     'EPOCH': e+1, 
            #     'UPDATE': (e + 1) *len(train_data)})   
            w.update({'epoch loss': loss.item(), 
                'epoch accuracy': accuracy(full_full_labels, full_full_output),
                'epoch percent error': percentError(full_full_output, full_full_labels),
                'label': labels.flatten()[0],
                'output': outputs.flatten()[0]})
            wandb.log(
                # {
                #     'loss': loss.item(), 
                #     # 'params': [p for p in net.parameters()], 
                #     # 'grads': [p.grad for p in net.parameters()], 
                #     # 'inputs': inputs,
                #     f'label {i}': labels.flatten()[i] for i in len(labels.flatten().size(0)),
                #     f'output {i}': outputs.flatten()[i] for i in len(outputs.flatten().size(0)),
                #     'accuracy': accuracy(labels, outputs)
                # }
                w
            )     
            if w['epoch accuracy'] == 1:

                PATH = f'.accuracynet.pth'
                torch.save(net.state_dict(), PATH)
            if w['epoch PR_AUC'] == 1:

                PATH = f'.PRnet.pth'
                torch.save(net.state_dict(), PATH)
            if w['epoch ROC_AUC'] == 1:

                PATH = f'.ROCnet.pth'
                torch.save(net.state_dict(), PATH)


            # wandb.log(wandlog) 

            full_output = None
            full_full_output = None
            running_loss = 0
            correct = 0
            total = 0           
            running_loss = 0
            net.eval()
            correct = 0
            total = 0
            if e % 10 == 9:
                PATH = f'.net.pth'
                torch.save(net.state_dict(), PATH)
            #check val set
            for i, data in enumerate(val_data, 0):
                inputs, labels = data[0], data[1].to(device) 
                outputs = net(inputs)
                loss = criterion(outputs, labels) 
                loss.backward()
                running_loss += loss.item()
                total+= labels.size(0)   
                if full_output == None:
                    full_output = outputs
                else:
                    full_output = torch.cat((full_output, outputs), 0)  
            # if e % 300 == 299:
            print(f'Validation loss for Epoch [{e +1}]: {running_loss/total}') 
            # if logpath != None:
            #     f.write(f'Validation loss for Epoch [{epoch}]: {running_loss/total}')  
            
            # wl = roc(full_val_labels, full_output)
            wandlog = {}
            # for q in wl:
            #     wandlog[f"{q} ROC_AUC"] = wl[q][0]
            #     wandlog[f"{q} PR_AUC"] = wl[q][1]
            #     wandlog[f"{q} ACCURACY"] = wl[q][2]
            # wandlog.update({
            #     "LOSS": running_loss / len(val_data), 
            #     "TYPE": "VAL", 
            #     'EPOCH': e+1, 
            #     'UPDATE': (e + 1)*len(train_data)})           
            # wandb.log(wandlog) 
            # best = min(best, running_loss / total)
            # early stop just goes to the next lr change checkpoint
           
            if bestval <= running_loss / total:
                # if epoch >= 5:
                #     print('Early Stop')
                #     print(f"Best Validation loss: {bestval}")
                #     print(f"Current Validation loss: {running_loss / total}")
                    
                e = e
                #     break
                # continue
                # return
            else:
                # bestnet.load_state_dict(copy.deepcopy(net.state_dict()))
                bestepoch = e
                bestval = running_loss / total

            running_loss = 0
            correct = 0
            total = 0
            times.append(time.monotonic() - ftime)
            PATH = f'.net.pth'
            torch.save(net.state_dict(), PATH)
            # if e % 300 == 299:
            print('time for epoch: ', times[-1], 'seconds')
            if logpath != None:
                f.write(f'time for epoch: {times[-1]}, seconds') 
            e+=1
        




    # finish training. in future dont plot and save here just return them
    print('Finished Training')
    print('average time per epoch: ', sum(times)/len(times), 'seconds')
    if logpath != None:
            f.write('Finished Training')
            f.write(f'average time per epoch: {sum(times)/len(times)} seconds')
            f.close()
    
    return 


# def compute_dataframe(df: pd.DataFrame, checkpoint_path: str):
#     model = LogisticRegression().load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
#     targets, targets_s, guides, guides_s = df.target_sequence.tolist(),  df.target_strand.tolist(), df.grna_target_sequence.tolist(),  df.grna_target_strand.tolist()
#     preds = []
#     for guide, target, guide_s, target_s in zip(guides, targets, guides_s, targets_s):
#         pred = model([one_hot(guide, guide_s), one_hot(target, target_s)])
#         preds.append(pred.item())
#     df['pred'] = preds
#     return df

def compute_dataframe(df: pd.DataFrame, checkpoint_path):
    model = checkpoint_path
    targets, targets_s, guides, guides_s = df.target_sequence.tolist(),  df.target_strand.tolist(), df.grna_target_sequence.tolist(),  df.grna_target_strand.tolist()
    preds = []
    for guide, target, guide_s, target_s in zip(guides, targets, guides_s, targets_s):
        pred = model([one_hot(guide, guide_s), one_hot(target, target_s)])
        preds.append(pred.item())
    df['pred'] = preds
    return df