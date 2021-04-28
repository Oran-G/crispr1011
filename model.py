
import torch.nn as nn
import torch.nn.functional as F
import torch
# import pytorch_lightning as pl

if __name__ == "__main__":
    pass

class Net(nn.Module):

    def Layer(self, layer_count, channels, channels_in, stride, kernal=[45, 7], padding=[23, 3]):
        return nn.Sequential(
            Block(channels_in, channels, stride, kernal=kernal, padding=padding),
            *[Block(channels, channels, kernal=kernal, padding=padding) for _ in range(layer_count-1)])
    def __init__(self, n=4):
            super(Net, self).__init__()
            
            # self.encoder = nn.Linear(4, 16)
            self.pre1 = self.Layer(n, channels=6, channels_in=1, stride=1, kernal=[5, 7], padding=[2, 3])
            self.pre2 = self.Layer(n, channels=16, channels_in=6, stride=1, kernal=[11, 7], padding=[5, 3])
            self.pre3 = self.Layer(n, channels=32, channels_in=16, stride=1, kernal=[45, 7], padding=[22, 3])
            self.conv1 = self.Layer(n, channels=64, channels_in=32, stride=2, kernal=[45, 7], padding=[22, 3])
            self.conv2 = self.Layer(n, channels=128, channels_in=64, stride=2, kernal=[45, 7], padding=[22, 3])
            self.conv3 = self.Layer(n, channels=256, channels_in=128, stride=2, kernal=[45, 7], padding=[22, 3])
            # self.conv3 = self.Layer(n, channels=512, channels_in=256, stride=1, kernal=[45, 7], padding=[22, 3])
            self.pool = nn.MaxPool2d([6, 4])
            self.linear = nn.Linear(256, 1)





    def forward(self, x):
        guide = x[0]
        target = x[1]
        guide = F.relu(self.pre1(guide))
        guide = F.relu(self.pre2(guide))
        guide = F.relu(self.pre3(guide))
        target = F.relu(self.pre1(target))

        target = F.relu(self.pre2(target))
        target = F.relu(self.pre3(target))

        out = torch.cat((guide, target), dim=2)
        out = self.conv1(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        try:
            out = self.linear(out)
        except RuntimeError:
            print(x[1].size())
        return torch.sigmoid(out)




    



class Block(nn.Module):

    def __init__(self, channels_in, num_filters, stride=1, kernal=[45, 7], padding=[22, 3]):
        super(Block, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            self.projection = IdentityPadding(num_filters, channels_in, stride)

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=kernal, stride=[stride, 1], padding=padding)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernal, stride=1, padding=padding)
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
        x = self.relu2(x)
        return x




class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.i = nn.MaxPool2d(1, stride=[stride, 1])
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.i(out)
        return out


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.layer = nn.Linear(184, 1)
        

    def forward(self, x):
        guide = x[0]
        target = x[1]
        out = torch.cat((guide, target), dim=2)
        out = out.view(out.size(0), -1)
        out = self.layer(out)
        # print(net.layer.weights, net.layer.bias, x, out)
        return out

class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.layer = nn.Linear(184, 1)

    def forward(self, x):
        guide = x[0]
        target = x[1]
        out = torch.cat((guide, target), dim=2)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.layer(out))
        
        return out



class basicRNN(nn.Module):

    def __init__(self, device):
        super(basicRNN, self).__init__()
        self.layer = nn.Linear(14, 6)
        self.out = nn.Linear(6, 1)
        self.device = device
        # self.to(self.device)
    
    def forward(self, x):
        full = torch.cat((x[0], x[1]), dim=1)
        hidden = torch.zeros((1, 1, 6)).type_as(x)
        for i in range(list(full.shape)[1]):
            hidden = self.layer(torch.cat((full[i], hidden), dim=1))

        return self.out(hidden)




import pytorch_lightning as pl


class NewNet(pl.LightningModule):

    def Layer(self, layer_count, channels, channels_in, stride, kernal=[45, 7], padding=[23, 3]):
        return nn.Sequential(
            Block(channels_in, channels, stride, kernal=kernal, padding=padding),
            *[Block(channels, channels, kernal=kernal, padding=padding) for _ in range(layer_count-1)])
    def __init__(self):
            super(NewNet, self).__init__()
            
            # self.encoder = nn.Linear(4, 16)
            n = 1
            self.pre1 = self.Layer(n, channels=6, channels_in=1, stride=1, kernal=[5, 7], padding=[2, 3])
            self.pre2 = self.Layer(n, channels=16, channels_in=6, stride=1, kernal=[11, 7], padding=[5, 3])
            self.pre3 = self.Layer(n, channels=32, channels_in=16, stride=1, kernal=[45, 7], padding=[22, 3])
            self.conv1 = self.Layer(n, channels=64, channels_in=32, stride=2, kernal=[45, 16], padding=[22, 7])
            self.conv2 = self.Layer(n, channels=128, channels_in=64, stride=2, kernal=[45, 16], padding=[22, 7])
            self.conv3 = self.Layer(n, channels=256, channels_in=128, stride=2, kernal=[45, 16], padding=[22, 7])
            # self.conv3 = self.Layer(n, channels=512, channels_in=256, stride=1, kernal=[45, 7], padding=[22, 3])
            self.pool = nn.MaxPool2d([6, 8])
            self.linear = nn.Linear(256, 1)





    def forward(self, x):
        guide = x[0]
        target = x[1]
        guide = F.relu(self.pre1(guide))
        guide = F.relu(self.pre2(guide))
        guide = F.relu(self.pre3(guide))
        target = F.relu(self.pre1(target))

        target = F.relu(self.pre2(target))
        target = F.relu(self.pre3(target))

        out = torch.cat((guide, target), dim=3)
        out = self.conv1(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        try:
            out = self.linear(out)
        except RuntimeError:
            print(x[1].size())
        return torch.sigmoid(out)




    