"""
Willem Atack/Elijah French
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class SimulatorGAN(nn.Module):
    
    def __init__(self, activation = 'relu'):
        super(SimulatorGAN, self).__init__()

        self.tan = nn.Tanh()        
        self.sig = nn.Sigmoid()

        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation =='relu':
            self.g = nn.ReLU()
        elif activation == 'elu':
            self.g = nn.ELU()
        
        self.conv_layers = nn.Sequential(
            #encoder
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),
            nn.BatchNorm2d(16),
            self.g,
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 3, padding = 0),
            nn.BatchNorm2d(32),
            self.g,
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 3, padding = 0),
            nn.BatchNorm2d(64),
            self.g,

            #decoder
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 3, padding = 0),
            nn.BatchNorm2d(32),
            self.g,
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 5, stride = 3, padding = 0),
            nn.BatchNorm2d(16),
            self.g,
            nn.ConvTranspose2d(in_channels = 16, out_channels = 2, kernel_size = 5, stride = 1, padding = 0),
            self.sig
        )

    def forward(self, x):
    
        return self.conv_layers(x)