import torch
from torch import nn

class Conv2(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Two times conv2d + BN + ReLU

        [TODO:description]

        Parameters
        ----------
        in_features : int
            input features
        out_features : int
            output features
        """
        super(Conv2, self).__init__()
        kernel = 3

        self.layers = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
                nn.Conv2d(out_features, out_features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class CNN(nn.Module):
    def __init__(self):
        """
        U-Net-like convolutional neural network.

        [TODO:description]
        """
        super(CNN, self).__init__()

        self.descend1 = Conv2(1, 32)
        self.descend2 = Conv2(32, 64)
        self.bottom = Conv2(64, 64)
        self.ascend1 = Conv2(128, 32)
        self.ascend2 = Conv2(64, 32)
        self.last = nn.Conv2d(32, 1, 1, padding='same',
                padding_mode='reflect')
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x0):
        x1 = self.descend1(x0)
        x = self.pool(x1)

        x2 = self.descend2(x)
        x = self.pool(x2)

        x = self.bottom(x)
        x = self.upsample(x)

        x = torch.cat((x2,x),1)
        x = self.ascend1(x)
        x = self.upsample(x)

        x = torch.cat((x1, x),1)
        x = self.ascend2(x)
        
        out = x0 + self.last(x)

        return out
