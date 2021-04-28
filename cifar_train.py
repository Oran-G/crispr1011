print("importing torch")
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
from net import Net
from datetime import datetime
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import os

import copy



def train(batch, learning_rate, epochs, batch_per, path='./data'):
    #create a folder for graohs
    folder = str(datetime.now())
    
    os.mkdir(f'./graphs/{folder}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    workers = 4 if torch.cuda.is_available() else 0
    #get, prepare, and split data
    transform1 = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)])    
    print("importing files")
    
    loader1 = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(path, train=True, download=True,
            transform=transforms.Compose([transform2, transform1])), 
        batch_size=batch,
        sampler=ChunkSampler(45000))
    loader2 = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(path, train=True, download=True,
            transform=transform1), batch_size=batch,
        sampler=ChunkSampler(5000, start=45000))



    # loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=("./data" if device == "cpu" else "./data1/train"), train=True,
    #         download=True), 
    #     batch_size=batch, shuffle=True, num_workers=workers)
    # trainloader = []
    # valloader = []
    # loader = trainloader
    # for i, data in enumerate(loader, 0):
    #     if data[0].size()[0] == batch:
    #         if i >= len(loader) - (len(loader) / 10):
    #             data = transform1(data)
    #             valloader.append(data)
    #         else:    
    #             t = transforms.Compose([transform2, transform1])
    #             print(t, "hi")
    #             data = t(data)
    #             trainloader.append(data)

    trainloader = loader1
    valloader = loader2
        


    net = Net()
    net.to(device)
    #def optim, loss, and init graph data
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.875, weight_decay=0.00125)
    print("begin training")
    x = []
    y = []
    valx = []
    valy = []
    corx = []
    cory = []
    corvaly = []

    #these go down, and random loss is ~2.303 so 15 will be replaced
    best = 15
    bestval = 15

    for epoch in range(epochs):  # loop over the dataset multiple times
        if epoch == 30:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate/10, momentum=0.9, weight_decay=0.0001)
            print("change 1")
            net = copy.deepcopy((bestnet))  
        if epoch == 45:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate/100, momentum=0.9, weight_decay=0.0001)  
            print("change 2")  
            net = copy.deepcopy((bestnet))  
        if epoch == 50:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate/1000, momentum=0.9, weight_decay=0.001)  
            print("change 3")   
            net = copy.deepcopy((bestnet))  
        if epoch + 1 % 10 == 0:
            torch.save(bestnet.state_dict(), f'.cifar_net.pth')

        correct = 0
        total = 0
        running_loss = 0.0
        # train mode
        net.train()
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data[0].to(device), data[1].to(device) 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics
            running_loss += loss.item()
            if i % batch_per == batch_per - 1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per))
                    x.append((epoch * len(trainloader)) + i)
                    y.append(running_loss/batch_per)
                        
                    best = min(best, running_loss / batch_per)
                    running_loss = 0

                    print('Accuracy of the network on the ' + str(batch_per) + 'th update: %d %%' % (
                        100 * correct / total))
                    cory.append(100 * correct / total)   
                    corx.append((epoch * len(trainloader)) + i) 
                    correct = 0
                    total = 0
        running_loss = 0
        net.eval()
        correct = 0
        total = 0
        #check val set
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device) 
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()        
        corvaly.append(correct / total * 100)        
        valx.append(((epoch + 1) * len(trainloader)))
        valy.append(running_loss/len(valloader))
        print(f"Val loss: {running_loss/len(valloader)}. Accuracy: {correct / total * 100}")
        if bestval > running_loss / len(valloader):
            bestnet = copy.deepcopy(net)
            bestval = running_loss / len(valloader)
            beste = epoch
            bestnet = copy.deepcopy(net)
        else:
            print("worse")    
        # if running_loss/len(valloader) - bestval > (running_loss/len(valloader)) / ((epoch + 1) * 1):
            
        #     plt.plot(x, y, label = "train")
        #     plt.plot(valx, valy, label = "valid")
        #     plt.legend()
        #     plt.ylabel('Running Loss')
        #     plt.xlabel('Updates')
        #     plt.savefig(f'./graphs/{folder}/loss.png')  
        #     plt.clf()   

        #     plt.plot(corx, cory, label = "train")
        #     plt.plot(valx, corvaly, label = "valid")
        #     plt.legend()
        #     plt.ylabel('Accuracy')
        #     plt.xlabel('Updates')
        #     plt.savefig(f'./graphs/{folder}/accuracy.png') 
        #     print(beste)                     
        #     return bestnet
        #     PATH = f'.cifar_net.pth'
        #     torch.save(net.state_dict(), PATH)
        running_loss = 0
        correct = 0
        total = 0
        





    # finish training. in future dont plot and save here just return them
    print('Finished Training')
    plt.plot(x, y, label = "train")
    plt.plot(valx, valy, label = "valid")
    plt.legend()
    plt.ylabel('Running Loss')
    plt.xlabel('Updates')
    plt.savefig(f'./graphs/{folder}/loss.png')
    plt.clf()  
    plt.plot(corx, cory, label = "train")
    plt.plot(valx, corvaly, label = "valid")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Updates')
    plt.savefig(f'./graphs/{folder}/accuracy.png')   
    print(beste) 
    return bestnet
    PATH = f'.cifar_net.pth'
    torch.save(net.state_dict(), PATH)






class ChunkSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples
