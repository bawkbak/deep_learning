import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
import random
import numpy as np
import math
import argparse
import time
from torch.nn import init
import os
from dataloader import *
from torch.utils.data import Dataset, DataLoader
from model import *
from evaluator import *
from util import save_image
import wandb

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--data', default='data', type=str)
parser.add_argument('--store', default='model', type=str)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--cycle', default=5, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=500, type=int)
args = parser.parse_args()
print(args)

train_datasets = ICLEVRLoader(args.data, cond=True)
train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, shuffle = True,drop_last=True)

train_datasets_switch = ICLEVRLoader(args.data, cond=True)

test_datasets = ICLEVRLoader(args.data, cond=True, mode='test')
test_dataloader = DataLoader(test_datasets, batch_size = len(test_datasets), shuffle = False,drop_last=True)



Eval = evaluation_model()

netG = Generator().cuda()
netD = Discriminator2().cuda()
netG.weight_init(mean=0,std=0.02)
netD.weight_init(mean=0,std=0.02)

optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.99))

criterion = torch.nn.BCELoss().cuda()


def gen_z(num):
    tmp = torch.zeros((num, 200, 1, 1))
    z = torch.randn_like(tmp).cuda()

    return z



z_test = gen_z(len(test_datasets))

for epoch in range(args.epochs):
    loss_d = []
    loss_g = []
    score = 0
    
    
    for i_batch, sampled_data in enumerate(train_dataloader):
        netG.train()
        netD.train()
        # --------------------------------------- Setup ------------------------------------ # 
        inputs = sampled_data['Image'].cuda() # shape batch 3 64 64 ( h w ) , x
        conds = sampled_data['cond'].cuda() # shape batch 1 24

        train_dataloader_switch = DataLoader(train_datasets_switch, batch_size = inputs.shape[0], shuffle = True)
        inputs_head = next(iter(train_dataloader_switch))['Image'].cuda() # shape batch 3 64 64 ( h w ) , 
        
        z = gen_z(inputs.shape[0]) # latent : shape batch 100 1 1 ( h w ) , z
        label_real = torch.ones((inputs.shape[0],1)).cuda() # Real label
        label_fake = torch.zeros((inputs.shape[0],1)).cuda() # Fake label

        # ---------------------------------------Training Discriminator ------------------------------------ #
        optimizer_D.zero_grad()
        fake_img = netG(z, conds) # G(c, z) : x~
        
        d_real = netD(inputs, conds.float()) # D(c, x)
        d_fake = netD(fake_img.detach(), conds.float()) # D(c, x~)
        d_real_head = netD(inputs_head, conds.float()) # D(c, x^)
        
        # Discriminator loss
        d_real_loss = criterion(d_real, label_real)
        d_fake_loss = criterion(d_fake, label_fake)
        d_real_head_loss = criterion(d_real_head, label_fake)

        D_loss = 0.1*d_real_loss + 0.1*d_fake_loss + 0.8*d_real_head_loss


        loss_d.append(D_loss.item())
        D_loss.backward()
        optimizer_D.step()
        
        # --------------------------------------- Training Generator ------------------------------------ #
        label_real = 0.9*label_real
        for _ in range(args.cycle):
            optimizer_G.zero_grad()

            z = gen_z(inputs.shape[0]) # latent
            fake_img2 = netG(z, conds) # G(c, z)
            d_fake2 = netD(fake_img2, conds.float()) # D(G(c, z))

            # Generator loss
            G_loss = criterion(d_fake2, label_real)
            G_loss.backward()
            optimizer_G.step()

        loss_g.append(G_loss.item())
        print('Epoch : '+str(epoch+1)+' '+str(i_batch)+'/'+str(int(len(train_datasets)/args.batch_size)+1)+' || '+str(round(D_loss.item(),2))+' / '+str(round(G_loss.item(),2)),end='\r')

    # Testing
    netG.eval()
    netD.eval()
    with torch.no_grad():
        conds_test = next(iter(test_dataloader))['cond'].cuda()
        gen_img = netG(z_test, conds_test)
        conds_test = torch.squeeze(conds_test, -1)
        conds_test = torch.squeeze(conds_test, -1)
        score = Eval.eval(gen_img, conds_test)


    
    # wandb.log({"D_Loss": sum(loss_d)/len(loss_d)})
    # wandb.log({"G_loss": sum(loss_g)/len(loss_g)})
    # wandb.log({"Score": score})


    print('Epoch : ',epoch+1,' D loss : ',sum(loss_d)/len(loss_d),' G loss : ',sum(loss_g)/len(loss_g), ' Score : ', round(score,2))

    if (score>=0.65):
        save_image(gen_img, os.path.join(args.store, f'2epoch{epoch+1}.png'), nrow=8, normalize=True)
        torch.save(netD.state_dict(), args.store+'Discriminator_'+str(epoch+1)+'_'+str(score)+'_'+'.pth')
        torch.save(netG.state_dict(), args.store+'Generator_'+str(epoch+1)+'_'+str(score)+'_'+'.pth')