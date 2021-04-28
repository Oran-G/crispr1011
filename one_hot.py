import csv
import torch
import copy


def one_hot_encoder(data, data_name=None, label_name=None):
    newdata = list()
    for line in data:
        sequence = None
        sins = None
        for n in (line[data_name] if data_name is not None else line[0]):

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
        newdata.append([sequence, (line[label_name] if data_name is not None else line[1])])  
    return newdata      

# data = [
#     {'a': "agtca", 'b': 1.021}
# ]
# print(one_hot_encoder(data, 'a', 'b'))
# data = [
#     ["agtca",  1.021]
# ]
# print(one_hot_encoder(data))
        
def one_hot(data, sign='+'):
    sins = None
    sequence = None
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
    if len(data) != 23:
        for i in range(23 - len(data)):
            sequence = torch.cat((sequence, torch.zeros((1, 4))), dim=0)
    if sign == '-':
        sequence = torch.flip(sequence, [1])      
    return sequence   

# print(one_hot("agtca"))
# print(one_hot('agtca', '-'))
     


from model import Net

data = [one_hot("agtcagactacgtatgctaggct"), one_hot("aaaggttccaggtccgattggcc")]
data[0].unsqueeze_(0).unsqueeze_(0)
print(data[0].size())
data[1].unsqueeze_(0).unsqueeze_(0)
net = Net()
print(net(data))
        
