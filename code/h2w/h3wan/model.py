import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class Residual3DBlock(nn.Module):
    def __init__(self, channels=256, temporal=False):
        super().__init__()
        k=(3,3,3) if temporal else (1,3,3)
        p=(1,1,1) if temporal else (0,1,1)
        self.net=nn.Sequential(nn.GroupNorm(32,channels),nn.SiLU(),nn.Conv3d(channels,channels,k,padding=p),
                               nn.GroupNorm(32,channels),nn.SiLU(),nn.Conv3d(channels,channels,k,padding=p))
    def forward(self,x): return x+self.net(x)

class Mapper(nn.Module):
    def __init__(self, channels=256, checkpoint_blocks=True):
        super().__init__(); self.checkpoint_blocks=checkpoint_blocks
        self.stem=nn.Conv3d(24,channels,(1,3,3),padding=(0,1,1))
        self.low=nn.ModuleList([Residual3DBlock(channels,i%2==1) for i in range(4)])
        self.refine=nn.Sequential(nn.Conv3d(channels,channels,(1,3,3),padding=(0,1,1)),nn.SiLU(),
                                  nn.Conv3d(channels,channels,(1,3,3),padding=(0,1,1)))
        self.high=nn.ModuleList([Residual3DBlock(channels,i%2==1) for i in range(4)])
        self.head=nn.Conv3d(channels,16,(1,3,3),padding=(0,1,1))
    def blocks(self,x,blocks):
        for block in blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.checkpoint_blocks else block(x)
        return x
    def forward(self,x):
        assert x.shape[1]==24 and x.shape[2] in (5,7)
        b,c,t,h,w=x.shape
        x=self.blocks(self.stem(x),self.low)
        # Exactly nearest spatial-only x2, implemented without ANY temporal interpolation.
        x=x.repeat_interleave(2,dim=-2).repeat_interleave(2,dim=-1)
        x=self.head(self.blocks(self.refine(x),self.high))
        assert x.shape==(b,16,t,2*h,2*w)
        return x
