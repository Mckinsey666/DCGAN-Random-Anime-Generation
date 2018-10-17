# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:49:51 2018

@author: USER
"""

import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

import datasets
import GAN
import utils

if __name__ == '__main__':
    
    ########## Configuring stuff ##########
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))
    
    latent_dim = 100

    batch_size = 128
    steps = 100000
    d_update = 1
    g_update = 1
    smooth = 0.9
    
    config = ('batch_size-[{}]-'
              'steps-[{}]-'
              'd_updates-[{}]-'
              'g_updates-[{}]-'
              'smooth-[{}]').format( 
            batch_size, 
            steps, 
            d_update, 
            g_update,
            smooth)
    print('Configuration: {}'.format(config))
    
    
    root_dir = '../cropped'
    sample_dir = './samples/{}'.format(config)
    ckpt_dir = './checkpoints/{}'.format(config)
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    ########## Start Training ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])
    train_data = datasets.Anime(root_dir = root_dir,
                                 transform = transform)
    shuffler = datasets.Shuffler(dataset = train_data, batch_size = batch_size)
    
    
    G = GAN.Generator(latent_dim = latent_dim).to(device)
    D = GAN.Discriminator(latent_dim = latent_dim).to(device)
    G.apply(utils.weights_init)
    D.apply(utils.weights_init)    
    
    G_optim = optim.Adam(G.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    D_optim = optim.Adam(D.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    
    discrim_log = []
    gen_log = []
    criterion = torch.nn.BCELoss()
    
    for step_i in range(1, steps + 1):
        # Create real and fake labels (0/1)
        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
            
        ########## Training the Discriminator (for several times) ##########
        for _ in range(d_update):
            # We have samples of real images
            real_img = shuffler.get_batch().to(device)
            # Sample from a random distribution (N(0,1))
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_img = G(z).to(device)
            
                
            real_loss = criterion(D(real_img), soft_label)
            fake_loss = criterion(D(fake_img), fake_label)
                        
            # We want to maximize real score + fake score, hence we minimize the negation
            # Gradient descent on the negative
            discrim_loss = (real_loss + fake_loss)
                        
            D_optim.zero_grad()
            discrim_loss.backward()
            D_optim.step()
            
        ########## Training the Generator ##########
        for _ in range(g_update):
            # Sample from a random distribution (N(0,1))
        
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_img = G(z).to(device)
            
            gen_loss = criterion(D(fake_img), real_label)
                
            G_optim.zero_grad()
            gen_loss.backward()
            G_optim.step()
                
            ########## Updating logs ##########
            discrim_log.append(discrim_loss.item())
            gen_log.append(gen_loss.item())
            utils.show_process(steps, step_i, gen_log, discrim_log)
        ########## Checkpointing ##########
        
        if step_i == 1:
            save_image(utils.denorm(real_img[:64, :, :, :]), 
                           os.path.join(sample_dir, 'real.png'))
        if step_i % 500 == 0:
            save_image(utils.denorm(fake_img[:64, :, :, :]), 
                           os.path.join(sample_dir, 'fake_step_{}.png'.format(step_i)))
        if step_i % 2000 == 0:
            utils.save_model(G, G_optim, step_i, tuple(gen_log), 
                                     os.path.join(ckpt_dir, 'G.ckpt'.format(step_i)))
            utils.save_model(D, D_optim, step_i, tuple(discrim_log), 
                                     os.path.join(ckpt_dir, 'D.ckpt'.format(step_i)))
            utils.plot_loss(gen_log, discrim_log, os.path.join(ckpt_dir, 'loss.png'))

        

    
    
    