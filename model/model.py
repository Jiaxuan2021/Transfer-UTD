from .parts import *


class ResNet_UTD(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = head_block()
        self.backbone = backbone_block()
        self.dense = dense_block()

    def forward(self, x):
        x = self.head(x)
        x = self.backbone(x)
        x = self.dense(x)
        return x