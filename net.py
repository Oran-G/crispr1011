import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import copy

class Net(nn.Module):
    
    def __init__(self, n=9):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self.Layer(n, 16, 16, 1)
        self.layers2 = self.Layer(n, 32, 16, 2)
        self.layers3 = self.Layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)
        self.to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pad = nn.ZeroPad2d(2)
        # r = random.randint(-1000000, 100000000)
        random.seed(3686487634786389)
        # print(f"Net seed: {r}")
    def Layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            Block(channels_in, channels, stride),
            *[Block(channels, channels) for _ in range(layer_count-1)])
    
    def forward(self, x):
            
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    



class Block(nn.Module):
    
    def __init__(self, channels_in, num_filters, stride=1):
        super(Block, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            self.projection = IdentityPadding(num_filters, channels_in, stride)
        self.to("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(.01)

    def forward(self, x):
        oldx = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        if self.projection:
            oldx = self.projection(oldx)
        x += oldx
        # x = self.dropout(x)
        x = self.relu2(x)
        return x




class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.i = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
        self.to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.i(out)
        return out
