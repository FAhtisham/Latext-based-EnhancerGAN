import torch 
import torch.nn as nn 
import torch.nn.optim as optim 

import matplotlib.pyplot as plt 
import numpy as np  





class Generator(nn.Module): 
    def __init__(self,in_dims, out_dims):
        super().__init__()
        
        self.l1 = nn.Linear(in_dims, out_dims)
        self.l2 = nn.Linear(out_dims, out_dims)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.l2(out) 
        out = nn.ReLU(out) 
        return out
        
class Discriminator(nn.Module):
    def __init__ (self, in_dims, out_dims): 
        super().__init__() 
        
        self.l1 = nn.Linear(in_dims, out_dims)
        self.l2 = nn.Linear(out_dims, out_dims)
    
    def forward(self,x): 
        out = self.l1(x) 
        out = self.l2(x) 
        