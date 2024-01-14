import numpy as np
import copy
import tqdm
import torch
import os
import random
import time
import datetime
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from Memory import Buffer
from SimulatorGAN import Pix2Pix

class TrainGAN():

    ###Initialization

    def __init__(self, transitionmemory: Buffer, MODELDIRECTORY: str, LOADMODEL = False, lr = 0.005, membatch_size = 100):
        """Initialize the training class which involves an iteration method and a learning method
        """

        #Set variables to be used in the network and network learning
        self.MODELDIRECTORY = MODELDIRECTORY
        self.lr = lr
        

        #Initialize the memory
        self.transitionmemory = transitionmemory
        self.membatch_size = membatch_size
        

        #Load saved model or randomly initialize a model to be trained
        self.simulator = Pix2Pix()
        self.simulator = self.simulator.cuda()

        # Initialize two optimizers - one for discriminator and one for generator
        self.optimizer_disc = optim.Adam(self.simulator.patch_gan.parameters(), self.lr)
        self.optimizer_gen = optim.Adam(self.simulator.gen.parameters(), self.lr)

        if LOADMODEL:
            checkpoint = torch.load(self.MODELDIRECTORY + '/simulator.pt')
            self.simulator.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            

        #Start wandb and keep track of changes in hypers
        run = wandb.init(project = 'CC_OPM_Simulator', entity = 'elijahfrench2023',  config = {"LR": self.lr, "Batch Size": self.membatch_size, "Scheduler Stepsize": "n/a", "ModelType": "GAN"},
                          settings = wandb.Settings(code_dir="."))


    ###Iterating methods
    
    def iterate(self, n_iter = 100, n_plot = 50):
        """Run through OPM taking random actions and add states to memory 
        """

        #Track the network using wandb
        wandb.watch((self.simulator), log = 'all', log_freq = 100)

        for i in range(n_iter+1):
            
            self.learn()

            #Visualize network performance 
            if np.mod(i, n_plot) == 0:
                self.one_step_visualization()
                print(i, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            
                 
    ### Learning methods
        
    def learn(self):
        """Learn from the memory buffer using the DDPG algorithm if the buffer is of sufficient size
        """

        #Sample a minibatch of transitions from memory
        states, actions, state_s = self.transitionmemory.sample_buffer(self.membatch_size)

        action_grids = torch.zeros((self.membatch_size, 2, 60, 60))
        action_grids[:, 0, 30, 30] = actions[:, 0]
        action_grids[:, 1, 9, 49] = actions[:, 1]
        action_grids[:, 1, 49, 9] = actions[:, 1]
        state_action = torch.cat((action_grids, states), 1).cuda()

        self.optimizer_disc.zero_grad()
        self.optimizer_gen.zero_grad()

        # train the discriminator first
        train_mode = 0
        disc_loss = self.simulator.training_step(state_action, state_s, train_mode)
        wandb.log({"Discriminator": float(disc_loss)})

        disc_loss.backward()
        self.optimizer_disc.step()

        train_mode = 1
        gen_loss = self.simulator.training_step(state_action, state_s, train_mode)

        gen_loss.backward()
        self.optimizer_gen.step()
        wandb.log({"Generator": float(disc_loss)})



    ### EValuation Function
        
    def evaluate(self, n_samples):


        TESTDIRECTORY = os.getenv('SLURM_TMPDIR') + '/Transitions/Jan8'

        teststates = torch.load(TESTDIRECTORY + '/state.pt')
        teststate_s = torch.load(TESTDIRECTORY + '/state_.pt')
        testactions = torch.load(TESTDIRECTORY + '/action.pt')

        batch = torch.ones(int(testactions.size(dim=0))).multinomial(num_samples = n_samples, replacement = False)

        state = torch.squeeze(teststates[batch])
        state_ = torch.squeeze(teststate_s[batch])
        action = torch.squeeze(testactions[batch])

        action_grid = torch.zeros((n_samples, 2, 60, 60))
        action_grid[:, 0, 30, 30] = action[:, 0]
        action_grid[:, 1, 9, 49] = action[:, 1]
        action_grid[:, 1, 49, 9] = action[:, 1]
        state_action = torch.cat((action_grid, state), dim=1).cuda()
        state_ = state_.cuda()
        
        simulated = self.simulator.gen(state_action)

        rmse_pressure = self.L(state_[:,0,:,:], simulated[:,0,:,:]).item()
        rmse_saturation = self.L(state_[:,1,:,:], simulated[:,1,:,:]).item()

        return rmse_pressure, rmse_saturation




    def one_step_visualization(self):
        """Simulate a single step and use the plot method to show how well the simulator is doing. 
        """

        TESTDIRECTORY = os.getenv('SLURM_TMPDIR') + '/Transitions/Jan8'

        teststates = torch.load(TESTDIRECTORY + '/state.pt')
        teststate_s = torch.load(TESTDIRECTORY + '/state_.pt')
        testactions = torch.load(TESTDIRECTORY + '/action.pt')

        batch = torch.ones(int(testactions.size(dim=0))).multinomial(num_samples = 1, replacement = False)

        state = torch.squeeze(teststates[batch])
        state_ = torch.squeeze(teststate_s[batch])
        action = torch.squeeze(testactions[batch])

        self.grids(state, state_, action)

    # def episodic_visualization(self, pngbase, time_steps = 12):
    #     """Simulate an episode and use the plot method to show how well the simulator is doing.
    #     In this visualizer the states are fed back into itself to see the viability of the model
    #     as a replacement to OPM.  
    #     """

    #     #Reset environment (reset file to be used by flow)
    #     self.env.reset()

    #     #Define initial state
    #     state = torch.stack((torch.ones((60,60))*3500/self.max_pressure, torch.zeros((60,60))))
    #     action = torch.rand(2)
    #     state = self.env.step(action, state, 0)
    #     simulated = state

    #     #Start running through an episode
    #     for step in range(1, time_steps+1):

    #         #Take a random action
    #         action = torch.rand(2)

    #         #Step in the encironment and store the observed tuple
    #         state_ = self.env.step(action, state, step)

    #         action_grid = torch.zeros((2, 60, 60))
    #         action_grid[0, 30, 30] = action[0]
    #         action_grid[1,  9, 49] = action[1]
    #         action_grid[1, 49, 9] = action[1]
    #         state_action = torch.cat((action_grid, simulated), 0)
    #         simulated_ = self.simulator(state_action)

    #         #If this step is the one randomly chosen, plot the simulators performance
    #         self.grids(state, state_, action, simulated_, pngname = pngbase + str(step) + '.png')

    #         simulated = simulated_.clone()
    #         state = state_.clone()

    def grids(self, state, state_, action, wandb_plot = True, pngname = ''):
        """Compare a new state to what the simulator predicted.
        """

        action_grid = torch.zeros((2, 60, 60))
        action_grid[0, 30, 30] = action[0]
        action_grid[1,  9, 49] = action[1]
        action_grid[1, 49, 9] = action[1]
        state_action = torch.cat((action_grid, state), 0).cuda()
        state_action = state_action.reshape(1,-1,60,60)
        simulated = self.simulator.gen(state_action).cpu()

        picture = plt.figure()
        picture.set_figheight(10)
        picture.set_figwidth(10)

        plt.subplot(4,2,1)
        plt.title('Pressure Grid')
        plt.ylabel('Initial State')
        self.plot(state[0,:,:], rates = np.array(action), action = True)

        plt.subplot(4,2,2)
        plt.title('Carbon Saturation')
        self.plot(state[1,:,:], rates = np.array(action), action = True)

        plt.subplot(4,2,3)
        plt.ylabel('New State')
        self.plot(state_[0,:,:])

        plt.subplot(4,2,4)
        self.plot(state_[1,:,:])

        plt.subplot(4,2,5)
        plt.ylabel('Simulated New State')
        self.plot(simulated[0, 0,:,:])

        plt.subplot(4,2,6)
        self.plot(simulated[0, 1,:,:])

        plt.subplot(4,2,7)
        plt.ylabel('Difference')
        self.plot(torch.abs(state_[0,:,:] - simulated[0, 0,:,:]), color = 'plasma')
        plt.xlabel('MSE: ' + str(round(np.sqrt(self.L(state_[0,:,:], simulated[0, 0,:,:]).item()), 5)))

        plt.subplot(4,2,8)
        self.plot(torch.abs(state_[1,:,:] - simulated[0, 1,:,:]), color = 'plasma')
        plt.xlabel('MSE: ' + str(round(np.sqrt(self.L(state_[1,:,:], simulated[0, 1,:,:]).item()), 5)))

        plt.tight_layout()
        if wandb_plot:
            wandb.log({"Simulations": wandb.Image(plt)})
        else:
            plt.savefig(pngname, format = 'png')
        plt.close()

    def plot(self, grid, color = 'Greens', colorbar = True, action = False, rates = [0,0]):
        """Create a plot based on the grid given and the color map specified.
        """

        p = np.array(grid.detach())
        cmap = plt.get_cmap(color)
        extent = [0, p.shape[1], 0, p.shape[0]]

        plt.imshow(p, cmap=cmap, origin = 'lower', interpolation='nearest', extent = extent,  vmin = 0, vmax = 1)

        border = patches.Rectangle((extent[0], extent[2]), extent[1], extent[3], linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(border)

        if action:
            injection = plt.Circle((30, 30), radius=1, color='red', alpha=1)
            plt.gca().add_patch(injection)
            production1 = plt.Circle((10, 10), radius=1, color='blue', alpha=1)
            plt.gca().add_patch(production1)
            production1 = plt.Circle((50, 50), radius=1, color='blue', alpha=1)
            plt.gca().add_patch(production1)

            plt.text(33, 33, str(round(rates[0], 2)), ha='center', va='center', color='black')
            plt.text(13, 13, str(round(rates[1], 2)), ha='center', va='center', color='black')
            plt.text(53, 53, str(round(rates[1], 2)), ha='center', va='center', color='black')

        if colorbar:
            plt.colorbar()

        plt.xticks([])
        plt.yticks([])

   
