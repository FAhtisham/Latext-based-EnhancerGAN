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

from dataset import load
from models import Autoencoder

import argparse


device = ("cuda" if torch.cuda.is_available() else "cpu")
device

def train(epoch):
  model.train()
  train_loss=0
  for i,x in enumerate(train_loader):
    optimizer.zero_grad()
    x = x.to(device)
    _, logits= model(x)

    #print(logits.size(), x.size())
    loss= criterion(logits, x)
    train_loss+=loss.item()
    loss.backward()

    optimizer.step()

    if interval > 0 and i % interval ==0:
      print("epoch: ", epoch, " batch: ", batch_size*i,"/", len(train_loader.dataset), " loss:", loss.item())


  train_loss /= len(train_loader)
  print('(Train) Epoch: {} | loss {:.4f}'.format(epoch, train_loss))
  return train_loss
  
  
'''
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--dropout", type=int, default=0.2)
parser.add_argument("--embedding-dim", type=int, default=200)
parser.add_argument("e-hidden-dim", type=int, default=100)

args = parser.parse_args()

print(args)
'''

seed= 0
epochs= 5000
batch_size= 32
lr= 5e-04 #4
dropout= 0.1
embedding_dims=40
e_hidden_dims= 100

seq_length= 131
bottleneck_dims=100
interval=10

d_hidden_dims=600

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


train_loader, nuc_pairs= load(batch_size, seq_length)
model = Autoencoder( nuc_pairs.size(), embedding_dims, e_hidden_dims, bottleneck_dims, d_hidden_dims, seq_length, dropout).to(device)


criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=lr)


print("\n Overall arhitecture of the model:\n",model)





for i,x in enumerate(train_loader):
    optimizer.zero_grad()
    x = x.to(device)
    print(x[0])
    break





# best_loss = 2.9
# total_loss = []
# for epoch in range(epochs):
#   loss = train(epoch)
#   total_loss.append(loss)
  
#   if loss < best_loss:
#       best_loss= loss
#       print('saved')
#       torch.save(model.state_dict(), 'ae.th')
      
# plt.plot(total_loss)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig('ae_loss.png')