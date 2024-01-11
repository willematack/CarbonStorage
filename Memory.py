"""
@author: Elijah French
"""

import torch
import os

class Buffer():

    def __init__(self, TRANSITIONDIRECTORY):
        
        self.TEMPDIRECTORY = TRANSITIONDIRECTORY

        self.states = torch.load(self.TEMPDIRECTORY + '/state.pt')
        self.state_s = torch.load(self.TEMPDIRECTORY + '/state_.pt')
        self.actions = torch.load(self.TEMPDIRECTORY + '/action.pt')

        self.mem_size = int(self.actions.size(dim=0))

    def sample_buffer(self, batch_size):
        '''Randomly sample from the buffer
        '''

        batch = torch.ones(self.mem_size).multinomial(num_samples = batch_size, replacement = False)

        states = self.states[batch]
        state_s = self.state_s[batch]
        actions = self.actions[batch]

        return states, actions, state_s