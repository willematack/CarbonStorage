"""
Willem Atack/Elijah French
"""

## This file implements the Pix2Pix model, comprised of a U-net generator and PatchGAN discriminator

import torch
from torch import nn, optim


## Generator Class (U-net)
class UNET(nn.Module):
    
    def __init__(self, activation = 'relu'):
        super(UNET, self).__init__()

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

    def forward(self, x):
        c1 = self.g(nn.BatchNorm2d(64)(self.conv1(x)))
        c1 = self.g(nn.BatchNorm2d(64)(self.conv2(c1)))
        pool1 = self.mp1(c1)

        c2 = self.g(nn.BatchNorm2d(128)(self.conv3(pool1)))
        c2 = self.g(nn.BatchNorm2d(128)(self.conv4(c2)))
        pool2 = self.mp2(c2)

        c3 = self.g(nn.BatchNorm2d(256)(self.conv5(pool2)))
        c3 = self.g(nn.BatchNorm2d(256)(self.conv6(c3)))
        pool3 = self.mp3(c3)

        #Middle
        cm = self.g(nn.BatchNorm2d(512)(self.conv7(pool3)))
        cm = self.g(nn.BatchNorm2d(512)(self.conv8(cm)))

        #Decoder
        d3 = self.deconv1(cm)
        u3 = torch.cat([d3, c3], dim = 1)
        u3 = self.g(nn.BatchNorm2d(256)(self.conv9(u3)))

        d2 = self.deconv2(u3)
        u2 = torch.cat([d2, c2], dim = 1)
        u2 = self.g(nn.BatchNorm2d(128)(self.conv10(u2)))

        d1 = self.deconv3(u2)
        u1 = torch.cat([d1, c1], dim = 1)
        u1 = self.g(nn.BatchNorm2d(64)(self.conv11(u1)))

        output = self.sig(self.output(u1))
    
        return output


## Discriminator Class (PatchGAN)
    
class PatchGAN(nn.Module):
    
    def __init__(self, activation = 'relu'):
        super(PatchGAN, self).__init__()

        self.tan = nn.Tanh()        
        self.sig = nn.Sigmoid()

        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation =='relu':
            self.g = nn.ReLU()
        elif activation == 'elu':
            self.g = nn.ELU()

        self.conv1 = nn.Conv2d(6, 64, kernel_size = 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride=2, padding=1)
        self.conv_out = nn.Conv2d(256, 1, kernel_size =1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x, y):

        x = torch.cat([x, y], dim=1)

        x1 = self.g(self.bn1(self.conv1(x))) # 30x30
        x2 = self.g(self.bn2(self.conv2(x1))) # 15 x 15
        x3 = self.g(self.bn3(self.conv3(x2))) # 7 x 7

        out = self.conv_out(x3)

        return out
        

## Pix2Pix Class
    
class Pix2Pix(nn.Module):

    def __init__(self, lambda_recon=10):

        super(Pix2Pix, self).__init__()
        
        self.lambda_recon = lambda_recon
        
        self.gen = UNET()
        self.patch_gan = PatchGAN()

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step(self, real_states, original_states):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        sim_states = self.gen(original_states)
        disc_logits = self.patch_gan(sim_states, original_states)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(sim_states, real_states)
        lambda_recon = self.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_states, original_states):
        sim_states = self.gen(original_states).detach()
        sim_logits = self.patch_gan(sim_states, original_states)

        real_logits = self.patch_gan(real_states, original_states)

        sim_loss = self.adversarial_criterion(sim_logits, torch.zeros_like(sim_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + sim_loss) / 2

    def training_step(self, state_action, sim_state, optimizer_idx):
        real = sim_state
        condition = state_action

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
        return loss
