import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import numpy as np
import torch.autograd as autograd


from models import Autoencoder
from models import GanBlock, Generator, Critic
from dataset import load
import math

import matplotlib.pyplot as plt

def compute_grad_penalty(critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    if torch.cuda.is_available():
        alpha = alpha.to(args.device)
    sample = alpha*real_data + (1-alpha)*fake_data
    sample.requires_grad_(True)
    score = critic(sample)

    outputs = torch.FloatTensor(B, args.bottleneck_dims).fill_(1.0)
    outputs.requires_grad_(False)
    if torch.cuda.is_available():
        outputs = outputs.to(args.device)
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    #grads = grads.view(B, -1)
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty

def plot_losses(g_loss, c_loss ):    
    plt.plot(g_loss)
    plt.xlabel("Epochs")
    plt.ylabel("W")
    plt.title("Generator's Loss")
    plt.savefig("gloss.png")
    plt.clf()
    
    plt.plot(c_loss)
    plt.ticklabel_format(style = 'plain')
    plt.autoscale(tight=True)
    plt.ylabel("log W")
    plt.xlabel("Epochs")
    plt.title("Critic's Loss")
    plt.savefig("closs.png")
    plt.clf()
    
closs_trace= []
gloss_trace= []
epochs_trace= []
def train(epcoh):
    autoencoder.eval()
    generator.train()
    critic.train()
    
    critic_loss= 0.0
    generator_loss= 0.0
    g_batches= 0
    

    for i, x in enumerate(train_loader):
        
        if torch.cuda.is_available():
            x=x.to(args.device)
        
        
        c_optim.zero_grad()

        
        # noise= normal.Normal(0,1)
        # noise= noise.sample([x.size(0), args.bottleneck_dims])
        # noise = noise.type(torch.cuda.FloatTensor)
        
        noise = torch.from_numpy(np.random.normal(0, 1, (x.size(0),
                                 args.bottleneck_dims))).float()
        
        
        if torch.cuda.is_available():
            noise= noise.to(args.device)
            
        with torch.no_grad():
            z_real= autoencoder(x)[0]
        #print(z_real.size())
        z_fake= generator(noise)
        
        real_score= critic(z_real)
        fake_score= critic(z_fake)
        
        grad_penalty = compute_grad_penalty(critic, z_real.data, z_fake.data)

        c_loss= -torch.mean(real_score) + torch.mean(fake_score) + args.lambda_gp*grad_penalty 
        critic_loss += c_loss.item()
        
        c_loss.backward()
        c_optim.step()
        
        if i% args.n_critic == 0:
            g_batches +=1
            g_optim.zero_grad()
            
            fake_score = critic(generator(noise))
            
            g_loss= -torch.mean(fake_score) 
            generator_loss += g_loss.item()
            
            g_loss.backward()
            g_optim.step
        
        if args.interval > 0 and i % args.interval ==0:
            print("Epoch : ", epoch, "batch : ", args.batch_size * i, "/", len(train_loader.dataset), "g loss : ", g_loss.item(), "c loss : ", c_loss.item())
            
    generator_loss /=g_batches
    critic_loss /= len(train_loader)
    
    print("(Train)", "epoch : ", epoch, "G loss : ", generator_loss, "C loss : ", critic_loss )
    closs_trace.append(critic_loss)
    gloss_trace.append(generator_loss)
    
    plot_losses(gloss_trace, closs_trace)
    return (generator_loss, critic_loss)
        


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout-size', type=float, default=0.1)
    parser.add_argument('--seq-length', type=int, default=131)
    parser.add_argument('--lambda-gp', type=float, default=10)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--bottleneck-dims', type=int, default=100)
    parser.add_argument('--block-dims', type=int, default=100)
    parser.add_argument('--e-hidden-dims', type=int, default=100)
    parser.add_argument('--d-hidden-dims', type=int, default=600)
    parser.add_argument('--embedding-dims', type=int, default=40)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--nuc-pair-size', type=int, default=83)
    parser.add_argument('--num_layers',type=int, default=6)
    args= parser.parse_args()
    
    
    print(args)
    
    train_loader, nuc_pairs= load(args.batch_size, args.seq_length)

    
    autoencoder= Autoencoder(args.nuc_pair_size, args.embedding_dims, args.e_hidden_dims, args.bottleneck_dims, args.d_hidden_dims, args.seq_length, args.dropout_size).to(args.device)
    autoencoder.load_state_dict(torch.load('ae.th', map_location=lambda x,y: x))
    print("\nAuroencoder model:",autoencoder,"\n")
    
    
    generator= Generator( args.num_layers,args.block_dims)
    critic= Critic(args.num_layers,args.block_dims)
    
    g_optim= optim.Adam(generator.parameters(), lr=args.lr)
    c_optim= optim.Adam(critic.parameters(), lr=args.lr)
    
    if torch.cuda.is_available():
        autoencoder= autoencoder.to(args.device)
        critic= critic.to(args.device)
        generator= generator.to(args.device)
        
        
    
    
    best_loss= np.inf
    
    # print("Generator Model:", generator,"\n")
    # print("Critic Model:", critic,"\n")
    
    for epoch in range(args.epochs):
        g_loss, c_loss= train(epoch)
        
        loss= g_loss + c_loss
        
        if loss < best_loss:
            best_loss=loss
            torch.save(generator.state_dict(), 'generator.th')
            torch.save(critic.state_dict(), 'critic.th')
            print("Models Saved")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    