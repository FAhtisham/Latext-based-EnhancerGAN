import torch
import numpy as np
from models import Autoencoder
from dataset import Sequences

x = torch.tensor([75, 23, 21, 83, 26, 45, 61,  3, 40, 45, 22, 64, 63, 76, 55, 24, 12, 12,
        11, 45, 45, 15, 52, 68, 70, 24, 23, 76,  5, 76, 21, 76, 60, 76, 83,  2,
        22, 18, 83,  5, 43, 24, 61, 28, 13, 52, 19, 74, 48,  8,  1, 21, 47, 66,
        76, 27, 61,  5, 59,  5, 80, 69, 10,  1,  5, 82, 49, 23,  1, 22, 68, 48,
        59, 42, 21, 82, 26, 80, 64, 15, 65, 19, 19, 65, 12, 73, 63, 52, 74, 19,
        6, 75, 19, 28,  8, 69, 83, 23,  3,  3,  5, 60,  6, 64,  1, 43, 42, 70,
        76, 42, 17, 12,  2, 70, 64, 59, 42, 81, 31, 64, 82, 10,  1, 54, 31, 27,
        61, 40, 18, 81,  6])
        
z=x.to("cuda:3")
z=z.unsqueeze(0)
seed= 0
epochs= 4000
batch_size= 64
# lr= 5e-04 #4
lr= 2e-04 #4
dropout= 0.1
embedding_dims=40
e_hidden_dims= 100

seq_length= 131
bottleneck_dims=100
interval=10

d_hidden_dims=600


autoencoder = Autoencoder( 83, embedding_dims, e_hidden_dims, bottleneck_dims, d_hidden_dims, seq_length, dropout).to("cuda:3")

                              
autoencoder.load_state_dict(torch.load('ae.th', map_location=lambda x,y: x))
autoencoder.eval()
_,logits=autoencoder(z)
print(logits.size() )


print(logits.size())
# print(logits[0,:,1])

pred=[]
for i in range(logits.size()[2]):
    max= torch.argmax(logits[0,:,i])
    pred.append(max.item())

print("Original Sequence:\n", z)
print("Reconstructed Sequence\n",pred)
decoding= Sequences(131).nucleotides.decoding
recon_str_=''
print(type(pred))

for c in pred:
    recon_str_+=decoding.get(c)

print("reconstructed seq:\n",recon_str_)
xz=z.cpu().numpy()
print(xz)
xa=np.asarray(xz,dtype=np.int32)
str_=''
for i in range(xa.shape[1]):
    str_+=decoding.get(xa[0,i])

print("original sequence:\n",str_)
