# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch
import math
from torch import nn

import util


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torchttorchh.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




class ResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1):
        super(ResNetDense, self).__init__()
        
        self.residual = nn.Linear(input_size, hidden_size)
        
        layers = []
        for _ in range(nlayers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
                #nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        residual = self.residual(x)
        layer = self.layers(x)
        return residual + layer



class DenseNet(nn.Module):
    def __init__(self, log, noise_levels, x_dim=2,
                 hidden_dim=32, time_embed_dim=16,
                 nresnet = 4,
                 use_fp16=False, cond=False):
        super(DenseNet, self).__init__()

        self.cond = cond #condition model over reco level inputs
        self.noise_levels = noise_levels        

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim

        
        
        self.time_dense  = nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.LeakyReLU(),
            #torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            
        )
        
        self.inputs_dense = nn.Sequential(
            nn.Linear(self.x_dim if self.cond==False else 2*self.x_dim, self.hidden_dim),
            nn.LeakyReLU(),
            #nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        
        self.residual = nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        
        self.resnet_layers = nn.ModuleList([
            ResNetDense(self.hidden_dim, self.hidden_dim) for _ in range(nresnet)
        ])
        
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 2*self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2*self.hidden_dim, self.x_dim)
        )
        
    def forward(self, inputs,steps,cond=None):
        
        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == inputs.shape[0]
        
        embed = self.time_dense(timestep_embedding(t, self.time_embed_dim))
        if self.cond:
            x = torch.cat([inputs, cond], dim=1)
        else:
            x = inputs
        
        inputs_dense = self.inputs_dense(x)        
        residual = self.residual(inputs_dense+ embed)
        x = residual
        for layer in self.resnet_layers:
            x = layer(x)
            
        output = self.final_layer(x)
        output = output + inputs
        return output
