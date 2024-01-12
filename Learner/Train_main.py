"""
@author: Elijah French/Willem Atack
"""

#Import necessary packages and classes 

'''self.conv_layers = nn.Sequential(
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
        
        )
        )'''



import os
import torch
import datetime
import wandb
from datetime import datetime

import Memory
import SimulatorCNN
import SimulatorGAN

import Train

from SimulatorCNN import SimulatorCNN
from SimulatorGAN import SimulatorGAN
from Memory import Buffer
from Train import Train

#Login to wandb

wandb.login(key='a7f822756af06f4332c2c78d8399c6a2b7f856cc')

#Define file path with which to save transitions in (note a .pt file must already be created in the folder) 

TEMPDIRECTORY = os.getenv('SLURM_TMPDIR')

#Create the folder to save transitions to
MODELDIRECTORY = TEMPDIRECTORY + '/Model_2'

if not os.path.isdir(MODELDIRECTORY):
    os.mkdir(MODELDIRECTORY)

#Initialize the memory

memory = Buffer(TRANSITIONDIRECTORY = TEMPDIRECTORY + '/Transitions/Dec_20')

#Train the model
modeltype = "CNN"
model = Train(transitionmemory = memory, MODELDIRECTORY = MODELDIRECTORY, 
              LOADMODEL = False,  membatch_size = 128, lr = 0.005, modeltype = modeltype)

model.iterate(n_iter = 10000)

print("Model done training.")

print("Evaluate the model:")

mse_pressure, mse_saturation = model.evaluate(128)

print("Test MSE (pressure): ", mse_pressure)
print("Test MSE (saturation): ", mse_saturation)
print(model.simulator)

#Visualize the model

#model.episodic_visualization()

#Code to store model in debug console

torch.save({'model_state': model.simulator.state_dict(),
            'optimizer_state': model.optimizer.state_dict(),
            'scheduler_state': model.scheduler.state_dict()},
            MODELDIRECTORY + '/simulator.pt')
