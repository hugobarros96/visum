import torch.nn as nn
import numpy as np
import torch

from .coord_conv_pt import CoordConv

class UNet(nn.Module):
    def __init__(self, in_channels, n_filters, out_channels, batchnorm=True, coordconv=True):
        super(UNet, self).__init__()
        assert isinstance(coordconv, bool), "coordconv needs to be a boolean, True if using coordconv layer, False otherwise"
        assert isinstance(batchnorm, bool), "batch_norm needs to be a boolean, True if using batch normalization layer, False otherwise"
 
        self.max_pool1 = nn.MaxPool2d(2,2)
        self.max_pool2 = nn.MaxPool2d(2,2)
        self.max_pool3 = nn.MaxPool2d(2,2)
        self.max_pool4 = nn.MaxPool2d(2,2)
        if coordconv==True:
            self.conv_block1 = ConvBlock(in_channels, n_filters, coordconv=coordconv, batch_norm=batchnorm)
        else:
            self.conv_block1 = ConvBlock(in_channels, n_filters, batch_norm=batchnorm)
        
        self.conv_block2 = ConvBlock(n_filters,n_filters * 2, batch_norm=batchnorm )
        self.conv_block3 = ConvBlock(n_filters * 2,n_filters * 4, batch_norm=batchnorm)
        self.conv_block4 = ConvBlock(n_filters * 4,n_filters * 8, batch_norm=batchnorm)
        self.conv_block5 = ConvBlock(n_filters * 8,n_filters * 16, batch_norm=batchnorm)

        self.up_conv1 = UpConvBlock(n_filters*16, n_filters*8, batch_norm=batchnorm)
        self.up_conv2 = UpConvBlock(n_filters*8, n_filters*4, batch_norm=batchnorm)
        self.up_conv3 = UpConvBlock(n_filters*4, n_filters*2, batch_norm=batchnorm)
        self.up_conv4 = UpConvBlock(n_filters*2, n_filters, batch_norm=batchnorm)
    
        self.last_conv = nn.Conv2d(n_filters, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #Contracting path
        c1 = self.conv_block1(x)
        p2=self.max_pool1(c1)
        c2 = self.conv_block2(p2)
        p3=self.max_pool2(c2)
        c3 = self.conv_block3(p3)
        p4=self.max_pool3(c3)
        c4 = self.conv_block4(p4)
        p5=self.max_pool4(c4)
        c5 = self.conv_block5(p5)

        #Expansive path
        c6 = self.up_conv1(c5, c4)
        c7 = self.up_conv2(c6, c3)
        c8 = self.up_conv3(c7, c2)
        c9 = self.up_conv4(c8, c1)

        out= self.sigmoid(self.last_conv(c9))

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding=1, batch_norm=True, coordconv=False, dropout=0.2):
        super(ConvBlock, self).__init__()
        assert isinstance(coordconv, bool), "coordconv needs to be a boolean, True if using coordconv layer, False otherwise"
        assert isinstance(batch_norm, bool), "batch_norm needs to be a boolean, True if using batch normalization layer, False otherwise"

        block = []
        if coordconv:
            block.append(CoordConv(in_size, out_size, kernel_size=3, padding=1))
        else:
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, momentum=0.01, eps=1e-03))
        block.append(nn.Dropout(p=dropout))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, eps=1e-03, momentum=0.01))
        block.append(nn.Dropout(p=dropout))
        block.append(nn.ReLU())
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)

        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm):
        super(UpConvBlock, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.conv_block = ConvBlock(in_size, out_size, padding=1, batch_norm=batch_norm)

    def forward(self, tensor1, tensor2):
        tensor1 = self.transpose(tensor1)
        #tensor2 = self.center_crop(tensor2, tensor1.shape[2:])
        out = torch.cat([tensor1, tensor2], 1)

        out = self.conv_block(out)

        return out
