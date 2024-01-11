"""
@author: Elijah French/Willem Atack
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

'''self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            self.g,
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            self.g,
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(8),
            self.g,
            nn.Conv2d(in_channels = 8, out_channels = 5, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(5),
            self.g,
            nn.Conv2d(in_channels = 5, out_channels = 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            self.g,
            nn.Conv2d(in_channels = 4, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(3),
            self.g,
            nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = 3, stride = 1, padding = 1),
            self.sig
        )'''
        
       


class SimulatorCNN(nn.Module):
    
    def __init__(self, activation = 'relu'):
        super(SimulatorCNN, self).__init__()

        self.tan = nn.Tanh()        
        self.sig = nn.Sigmoid()

        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation =='relu':
            self.g = nn.ReLU()
        elif activation == 'elu':
            self.g = nn.ELU()
        
        # Encoder
        self.conv1 = nn.Conv2d(4, 64, (3,3), padding = "same")
        self.conv2 = nn.Conv2d(64, 64, (3,3), padding = "same")
        self.mp1 = nn.MaxPool2d((2,2))

        #30x30
        self.conv3 = nn.Conv2d(64, 128, (3,3), padding = "same")
        self.conv4 = nn.Conv2d(128, 128, (3,3), padding = "same")
        self.mp2 = nn.MaxPool2d((2,2))

        #15x15
        self.conv5 = nn.Conv2d(128, 256, (3,3), padding = "same")
        self.conv6 = nn.Conv2d(256, 256, (3,3), padding = "same")
        self.mp3 = nn.MaxPool2d((2,2), ceil_mode=True)

        #8x8 - Middle
        self.conv7 = nn.Conv2d(256, 512, (3,3), padding = "same")
        self.conv8 = nn.Conv2d(512, 512, (3,3), padding = "same")

        # Decoder - 16x16
        self.deconv1 = nn.ConvTranspose2d(512, 256, (3,3), stride = (2,2), padding = 1)
        self.conv9 = nn.Conv2d(512, 256, (3,3), padding = "same")

        self.deconv2 = nn.ConvTranspose2d(256, 128, (3,3), stride = (2,2), padding = 1, output_padding=1)
        self.conv10 = nn.Conv2d(256, 128, (3,3), padding = "same")

        self.deconv3 = nn.ConvTranspose2d(128, 64, (3,3), stride = (2,2), padding = 1, output_padding=1)
        self.conv11 = nn.Conv2d(128, 64, (3,3), padding = "same")

        self.output = nn.Conv2d(64, 2, (3,3), padding = "same")

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)

        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)

        self.bn9 = nn.BatchNorm2d(256)
        self.bn10 = nn.BatchNorm2d(128)
        self.bn11= nn.BatchNorm2d(64)


    def forward(self, x):
        c1 = self.g(self.bn1(self.conv1(x)))
        c1 = self.g(self.bn2(self.conv2(c1)))
        pool1 = self.mp1(c1)

        c2 = self.g(self.bn3(self.conv3(pool1)))
        c2 = self.g(self.bn4(self.conv4(c2)))
        pool2 = self.mp2(c2)

        c3 = self.g(self.bn5(self.conv5(pool2)))
        c3 = self.g(self.bn6(self.conv6(c3)))
        pool3 = self.mp3(c3)

        #Middle
        cm = self.g(self.bn7(self.conv7(pool3)))
        cm = self.g(self.bn8(self.conv8(cm)))

        #Decoder
        d3 = self.deconv1(cm)
        u3 = torch.cat([d3, c3], dim = 1)
        u3 = self.g(self.bn9(self.conv9(u3)))

        d2 = self.deconv2(u3)
        u2 = torch.cat([d2, c2], dim = 1)
        u2 = self.g(self.bn10(self.conv10(u2)))

        d1 = self.deconv3(u2)
        u1 = torch.cat([d1, c1], dim = 1)
        u1 = self.g(self.bn11(self.conv11(u1)))

        output = self.sig(self.output(u1))
    
        return output