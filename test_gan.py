import torch
import torch.nn as nn
import numpy as np

from model import Autoencoder, Generator
from dataset import Sequences

decoding= Sequences(131).nucleotides.decoding





epochs= 4000
batch_size= 64
lr= 5e-04 #4
dropout= 0.1
embedding_dims=40
e_hidden_dims= 100

seq_length= 131
bottleneck_dims=100
interval=10

d_hidden_dims=600


autoencoder = Autoencoder( 83, embedding_dims, e_hidden_dims, bottleneck_dims, d_hidden_dims, seq_length, dropout).to("cuda:3")
autoencoder.load_state_dict(torch.load('ae.th', map_location='cpu'))
autoencoder.eval()

generator= Generator(10,100)
generator.load_state_dict(torch.load('generator.th', map_location='cpu'))
generator.eval()

noise=torch.FloatTensor(np.random.normal(0,1,(1,100)))
z= generator(noise[None, :,:])
z=z.to("cuda:3")
print(noise.size())
logits= autoencoder.decoder(z).squeeze()
print(logits.size())

# gen_seq= logits.argmax(dim=0)
# # print(gen_seq)

pred=[]
for i in range(logits.size()[1]):
    max= torch.argmax(logits[:,i])
    pred.append(max.item())



decoding= Sequences(131).nucleotides.decoding
recon_str_=''
print(pred)
for c in pred:
    recon_str_+=decoding.get(c)
print(recon_str_)