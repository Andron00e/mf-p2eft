from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from definitions.networks.abx import TopK


class ResidualBlock(nn.Module):
    def __init__(self, out_features, block=nn.Linear, stride=2):
        super(ResidualBlock, self).__init__()

        self.seq = nn.Sequential(
            # {stride} N->N layers is one block
            *[block(out_features, out_features)]*stride  # yes, it will work
        )
        
    def forward(self, x):
        # {stride} layers and a skip connection over these is one block
        return self.seq(x) + x  


class LinResNet(nn.Module):
    def __init__(self, block, in_dim, num_blocks, lin=nn.Linear, num_features=512, num_classes=10, softmax=0):
        super(LinResNet, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Flatten(1),
            lin(in_dim, num_features),  # convert input to N
            nn.Sequential(*[block(num_features)]*num_blocks),  # {num_blocks} blocks N->N
            lin(num_features, num_classes),  # convert N to output
            *[nn.Softmax(dim=1)]*softmax,
        )
        
    def forward(self, x):
        return self.seq(x)


def topkresnet(
        in_dim=3*32*32, num_classes=10, num_blocks=3, T=0., k=None, r=0.30, f='ba', T_param=False,
        num_features=512, stride=2, softmax=0,
        ):
    topk_layer = partial(TopK, T=T, k=k, r=r, f=f, T_param=T_param)
    block = partial(ResidualBlock, block=topk_layer, stride=stride)
    return LinResNet(
        block=block, in_dim=in_dim, num_blocks=num_blocks, lin=topk_layer, num_features=num_features,
        num_classes=num_classes, softmax=softmax
    )
