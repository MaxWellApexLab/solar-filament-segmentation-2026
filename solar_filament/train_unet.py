"""Compact U-Net trained exclusively on organizers' annotated training images.

No external pretrained models or competing test labels; train/val must split by
physical observation/date, not by annotation id, to avoid multi-annotator leakage.
This is training infrastructure, not trained weights without the official files.
"""
import torch
from torch import nn
import numpy as np


def block(ni,no):
    return nn.Sequential(nn.Conv2d(ni,no,3,padding=1,bias=False),nn.BatchNorm2d(no),nn.SiLU(),
                         nn.Conv2d(no,no,3,padding=1,bias=False),nn.BatchNorm2d(no),nn.SiLU())

class FilamentUNet(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.e1=block(1,base);self.e2=block(base,base*2);self.e3=block(base*2,base*4)
        self.pool=nn.MaxPool2d(2)
        self.u2=block(base*6,base*2);self.u1=block(base*3,base)
        self.head=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(self.pool(a));c=self.e3(self.pool(b))
        y=self.u2(torch.cat((nn.functional.interpolate(c,size=b.shape[2:],mode='bilinear',align_corners=False),b),1))
        return self.head(self.u1(torch.cat((nn.functional.interpolate(y,size=a.shape[2:],mode='bilinear',align_corners=False),a),1)))

def combined_loss(logits,target):
    target=target.float()
    bce=nn.functional.binary_cross_entropy_with_logits(logits,target)
    probs=torch.sigmoid(logits)
    intersection=(probs*target).sum(dim=(2,3));total=probs.sum((2,3))+target.sum((2,3))
    return bce+(1-(2*intersection+1)/(total+1)).mean()
