# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:19:30 2018

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.latent_dim, 
                                       out_channels = 1024, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 1024,
                                       out_channels = 512,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 512,
                                       out_channels = 256,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 128,
                                       out_channels = 1,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1),
                    nn.Sigmoid()
                    )
        return
    
    def forward(self, input):
        return self.gen(input)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        
        self.discrim = nn.Sequential(
                    nn.Conv2d(in_channels = 1, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1),
                    nn.Sigmoid()
                    )

        return
    
    def forward(self, input):
        return self.discrim(input).view(-1)
    
    
if __name__ == '__main__':
    # Image Tensor shape: N * C * H * W
    # Batch size, channels, height, width respectively
    
    latent_dim = 100
    
    z = torch.randn(5, latent_dim, 1, 1)
    g = Generator(latent_dim = latent_dim)
    d = Discriminator(latent_dim = latent_dim)
    
    img = g(z)
    
    print(img.shape)
    print(d(img).shape)
    