import torch
from .parts import *
import sys
sys.path.append('..')

class Encode_Net(nn.Module):
    # encoder part, input 1 curve sequence(1 channel), output channel is 64
    def __init__(self):
        super(Encode_Net, self).__init__()
        self.inc = Conv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x3

class Decode_Net1(nn.Module):
    # input is 64 channels, output is 1 channel, estimated 'a'
    def __init__(self):
        super(Decode_Net1, self).__init__()
        self.up1 = Up1(64, 32)
        self.up2 = Up2(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6


class Decode_Net2(nn.Module):
    # input is 64 channels, output is 1 channel, estimated 'bb'
    def __init__(self):
        super(Decode_Net2, self).__init__()
        self.up1 = Up1(64, 32)
        self.up2 = Up2(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6
