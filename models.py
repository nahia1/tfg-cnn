import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        """
        U-Net-like convolutional neural network.

        [TODO:description]
        """
        super(CNN, self).__init__()
        kernel = 3
        features = 32

        self.first_layer = nn.Sequential(
                nn.Conv2d(1, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU(),
        )
        
        self.last_layer = nn.Sequential(
                nn.Conv2d(features, 1, 1, padding='same',
                    padding_mode='reflect'),
        )

        self.conv1 = nn.Sequential(
                nn.Conv2d(features, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU()
        )

        self.conv2 = nn.Sequential(
                nn.Conv2d(features*2, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU(),
                nn.Conv2d(features, features, kernel, padding='same',
                    padding_mode='reflect'),
                nn.BatchNorm2d(features),
                nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x0):
        x1 = self.first_layer(x0)
        x = self.pool(x1)

        x2 = self.conv1(x)
        x = self.pool(x2)

        x = self.conv1(x)
        x = self.upsample(x)

        x = torch.cat((x2,x),1)
        x = self.conv2(x)
        x = self.upsample(x)

        x = torch.cat((x1, x),1)
        x = self.conv2(x)
        
        out = x0 + self.last_layer(x)

        return out
