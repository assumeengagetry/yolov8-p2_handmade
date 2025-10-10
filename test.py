import math 
from typing import List , Tuple

import torch 
import torch.nn as nn 
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, c1, c2, k=3 , s=1,p= None  ,bias=False):
        super().__init__()
        if p is None 
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias= bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv))


class BottleNeck(nn.Module):
    def __init__(self ,c1 ,c2 ,shortcut = True , expaction = 0.5):
        super().__init__()
        c_ = int(c2 * expaction)
        self.conv1 = ConvBnAct(c1, c_ ,k=1)
        self.conv2 = ConvBnAct(c2, c_ ,k=3)
        self.use_add = shortcut and c1 == c2    
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            return x + y 
        return y



class C2f(nn.Module):
    def __init__(self, c1, c2, n = 1, expaction = 0.5):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2, k = 1)
        self.cv2 = cv2
        self.n = n 
        self.expand = expaction
        m = c2 //2 
        self.blocks == nn.ModuleList([BottleNeck(n ,m ,shortcut = True, expaction = 1.0) for _ in range(n)])


        self.cv2 = ConvBnAct(())


























