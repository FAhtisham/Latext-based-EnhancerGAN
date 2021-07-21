import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import collections

from tqdm import tqdm




# class Autoencoder(nn.Module):
#   def __init__(self, nuc_pair_size, embedding_dims, e_hidden_dims, bottleneck_dims, d_hidden_dims, seq_length, dropout_size = 0.1):
#     super().__init__()
#     nuc_pair_size+=1
#     self.seq_length= seq_length
#     # define the vars over here (layers, objects)
#     self.embedding= nn.Embedding( nuc_pair_size, embedding_dims)
#     self.rnn1= nn.LSTM(input_size= embedding_dims, hidden_size= e_hidden_dims)#, num_layers=2)#, bidirectional=True)
#     self.fc0= nn.Linear(in_features = e_hidden_dims, out_features= 200)
#     self.fc1= nn.Linear(in_features = 200, out_features= bottleneck_dims)
#     self.a1= nn.ReLU(True)
#     #self.a1= nn.Sigmoid(True)
#     self.dropout= nn.Dropout(dropout_size)
    
#     self.fc02= nn.Linear(in_features = bottleneck_dims, out_features= 200)
#     self.fc2= nn.Linear(in_features = 200, out_features= d_hidden_dims)
#     self.rnn2= nn.LSTM(input_size= d_hidden_dims, hidden_size= d_hidden_dims, num_layers=2)#, bidirectional=True)
#     self.fc3= nn.Linear(in_features= d_hidden_dims, out_features= nuc_pair_size)



#   def encoder(self, x):
#     print("encoder's x", x.size())
#     x= self.embedding(x).permute(1,0,2)
#     print("encoder's embedding", x.size())
#     _,(hidden_states, _)= self.rnn1(x)
#     print("encoder hidden ", hidden_states.size())
#     lv= self.fc0(hidden_states)
#     print("encoder's lv", lv.size())
#     lv= self.fc1(lv) # latent vector
#     print("encoder's lv", x.size())
#     lv= self.dropout(lv)
#     return lv

  
#   def decoder(self, lv):
#     # import pdb 
#     # pdb.set_trace()
#     print("dencoder's lv", lv.size())
#     lv=self.fc02(lv)
#     print("dencoder's lv2", lv.size())
#     lv= self.fc2(lv)
#     print("dencoder's lv3", lv.size())
#     #output, _= self.rnn2(lv.repeat(self.seq_length,1,1),(lv,lv))
#     output, _= self.rnn2(lv)
#     print("dencoder's output", output.size())
#     output= output.permute(1,0,2)
#     print("permute k bad",output.size())
#     logits= self.fc3(output)
#     print("dencoder's logits fc3", logits.size())
    
#     #print(logits.size())
#     logits = logits.transpose(1,2)
#     print("final logits: ",logits.size())
    
#     return torch.squeeze(logits)
  
#   def forward(self,x):
#     lv= self.encoder(x)
#     logits= self.decoder(lv)
#     return (lv.squeeze(), logits)
    
class Autoencoder(nn.Module):
    def __init__(self, nuc_pair_size, embedding_dims, e_hidden_dims, bottleneck_dims, d_hidden_dims, seq_length, dropout_size = 0):
        super().__init__()
        nuc_pair_size +=1
        self.seq_length= seq_length
        # define the vars over here (layers, objects)
        self.embedding= nn.Embedding( nuc_pair_size, embedding_dims)
        self.rnn1= nn.LSTM(input_size= embedding_dims, hidden_size= e_hidden_dims, num_layers= 1, bidirectional= True)
        self.fc1= nn.Linear(in_features = e_hidden_dims*2, out_features= bottleneck_dims)
        self.a1= nn.ReLU(True)
        self.dropout= nn.Dropout(dropout_size)
        self.fc2= nn.Linear(in_features = bottleneck_dims, out_features= d_hidden_dims)
        self.rnn2= nn.LSTM(input_size= d_hidden_dims, hidden_size= d_hidden_dims, num_layers=1, bidirectional=True)
        self.fc3= nn.Linear(in_features= d_hidden_dims*2, out_features= nuc_pair_size)
    
    def encoder(self, x):
        #print("encoder's x",x.size())
        x= self.embedding(x).permute(1,0,2)
        #print("encoder's x after emedding",x.size())
        _,(hidden_states, _)= self.rnn1(x)
        #print("encoder's hidd",hidden_states.size())
        # hidden_states = hidden_states[1]
        # hidden_states= hidden_states[None, :,:]
        
        hidden_states= torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim= 1)
        #print("mamamamam", hidden_states.size())
        lv= self.fc1(hidden_states) # latent vector
        # print("encoder's lv",lv.size())
        lv= self.dropout(lv)
        # print("encoder's lv",lv.size())
        return lv
    
    def decoder(self, lv):
        # import pdb
        # pdb.set_trace*()
        # print("dencoder's lv",lv.size())
        lv= self.fc2(lv)
        output, _= self.rnn2(lv.repeat(self.seq_length,1,1)) #,(lv,lv))
        #output=
        # print("dencoder's output rnn",output.size())
        output= output.permute(1,0,2)
        # print("dencoder's permute",output.size())
        logits= self.fc3(output)
        # print("dencoder's logits",logits.size())
        return logits.transpose(1,2)
    
    
    def forward(self,x):
        lv= self.encoder(x)
        logits= self.decoder(lv)
        return (lv.squeeze(), logits)   
    
    

class GanBlock(nn.Module):
    def __init__(self, block_dims):
        super().__init__()
    
        self.nnet= nn.Sequential(
        nn.Linear(block_dims, block_dims),
        nn.ReLU(True)
        #nn.Linear(block_dims, block_dims)        
        )
        
    def forward(self,x):
        return self.nnet(x) #+ x
      
class Generator(nn.Module):
    def __init__(self, n_layers, block_dims ):
        super().__init__()
        
        self.gnet= nn.Sequential(
        *[GanBlock(block_dims) for _ in range(n_layers)]
        )
      
    def forward(self,x):
        x= self.gnet(x)
        return x
        
class Critic(nn.Module):
    def __init__(self, n_layers, block_dims):
        super().__init__()
        
        self.cnet= nn.Sequential(*[GanBlock(block_dims) for _ in range(n_layers)]
        )
    
    def forward(self,x):
        x= self.cnet(x)
        return x