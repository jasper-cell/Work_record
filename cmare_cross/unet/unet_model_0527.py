""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 64)
        self.down2 = Down(64, 128)
        factor = 4 if bilinear else 1
        self.down3 = Down(128, 320)
        self.up1 = Up(448, 224, bilinear)
        self.up2 = Up(288, 144, bilinear)
        self.up3 = Up(160, 40, bilinear)
        self.outc = OutConv(40, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print('x1: ', x1.shape)
        x2 = self.down1(x1)
        #@print('x2: ', x2.shape)
        x3 = self.down2(x2)
        #print('x3:', x3.shape)
        x4 = self.down3(x3)
        #print('x4: ', x4.shape)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits