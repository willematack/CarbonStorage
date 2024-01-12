"""
Created on Wed June 14 17:06:12 2023

@author: Elijah French
@references: Kyubo Noh
"""

#Import necessary packages including os to interact with the terminal and files

import os
import torch
import datetime
import numpy as np
import torch.nn as nn
from datetime import datetime, timedelta

#Create environment class

class Environment:
    def __init__(self, OPMFILEDIRECTORY: str, OPMFILENAME: str, startdate = datetime(2023, 1, 1), time_steps = 12, OPMtextoutput = False, 
                 min_injection = 1_000, max_injection = 4_000, min_production = 1_000, max_production = 5_000, max_pressure_OPM = 10_000, 
                 dt = 'year'):
        
        #Change to relevant directory (where OPMFILENAME is stored)
        os.chdir(OPMFILEDIRECTORY)

        #Initialize necessary file names 
        self.OPMFILENAME = OPMFILENAME
        self.name = self.OPMFILENAME + 'COPY.DATA'
        self.realname = self.name[:-5]
        self.grid = self.name[:-5] + '.EGRID'
        self.output = self.name[:-5] + '.RSM'
        self.Sigmoid = nn.Sigmoid()

        #Create variables for conversion from actions to OPM schedule input
        self.min_injection = min_injection
        self.max_injection = max_injection
        self.min_production = min_production
        self.max_production = max_production
        self.time_steps = time_steps
        self.max_pressure = max_pressure_OPM
        self.max_pressure_OPM = max_pressure_OPM

        #Creates the list of dates to be used starting at the initial date (Jan 1, 2023)
        if dt == 'month':
            dlist = [startdate]
            for i in range(1, time_steps+1):
                next_month = startdate + timedelta(days=31 * i)  
                first_day_of_month = datetime(next_month.year, next_month.month, 1)
                dlist.append(first_day_of_month)
            self.dates = [t.strftime("%d '%b' %Y").upper()[1:] for t in dlist]
            self.delta_t = 30.4
        elif dt == 'year':
            self.dates = ["1 'JAN' " + str(2023 + i) for i in range(0, time_steps+1)]
            self.delta_t = 365.25
        elif dt == '2year':
            self.dates = ["1 'JAN' " + str(2023 + 2*i) for i in range(0, time_steps+2)]
            self.delta_t = 2*365.25

        #Define OPM command depending on whether text output is required
        if OPMtextoutput:
            self.command = 'flow ' + self.name + ' --enable-opm-rst-file=true'
        else:
            self.command = 'flow ' + self.name + ' --enable-opm-rst-file=true > /dev/null'

    def reset(self):
        '''Create a copy of the OPM input file for each episode/simulation
        '''
        file_copy = open(self.OPMFILENAME + '.DATA', 'r').readlines()
        with open(self.name, 'w') as copy_file:
            copy_file.writelines(file_copy)

    def restart_line(self, step):
        '''Add a line that ensures that OPM will create a restart file
        '''
        return "    '" + self.realname + "'  " +  str(step)  + "   1*  " + "   1*   /\n"
    
    def denormalize_injection(self, injection):
        '''Take output and ensure that OPM can take it as an input
        '''
        return injection*(self.max_injection - self.min_injection) + self.min_injection
    
    def denormalize_production(self, production):
        return production*(self.max_production - self.min_production) + self.min_production
    
    def action_to_schedule(self, action):
        '''Take an action and convert it into a line for OPM
        '''
        
        injection = self.denormalize_injection(action[0]).item()
        production1 = self.denormalize_production(action[1]).item()
        production2 = self.denormalize_production(action[1]).item()

        injection_action = '   Inj1	GAS	OPEN	RATE	' + str(round(injection)) + ' 1* 	10000 	1*	1*	1*/ \n'
        production_action1 = '   Prd1 	OPEN 	ORAT 	' + str(round(production1)) +  '	1* 	1* 	1*	1*	1000	0	0	0/ \n'
        production_action2 = '   Prd2 	OPEN 	ORAT 	' + str(round(production2)) +  '	1* 	1* 	1*	1*	1000	0	0	0/ \n'

        return [injection_action, production_action1, production_action2]

    def step(self, action, state, step): 
        '''Using the action step through the environment once
        '''

        #Denormalize step and action, turn action into schedule
        schedule = self.action_to_schedule(action)

        #Start modifying OPM file
        Startfile = open(self.name, "r")
        Startlines = Startfile.readlines()
        if step == 0:  
            #Set the starting date
            templine = Startlines.index('START\n')
            Startlines.insert(templine+1, self.dates[step] + ' /')
        elif step == 1:
            #Add RESTART keyword to solutions section 
            templine = Startlines.index('SOLUTION\n')
            Startlines.insert(templine+ 2, '\n')
            Startlines.insert(templine + 3, 'RESTART\n')
            Startlines.insert(templine + 4, self.restart_line(step)) 
            #Remove previous output from displaying
            templine = Startlines.index('SCHEDULE\n')
            Startlines.insert(templine + 2, '\n')
            Startlines.insert(templine + 3, 'SKIPREST\n')
            #Remove equilibriation
            templine1 = Startlines.index('EQUIL\n')
            templine2 = Startlines.index('SUMMARY\n')
            Startlines = Startlines[:templine1] + Startlines[templine2-1:]
        else:
            #Change timestep for RESTART keyword in solutions section
            templine = Startlines.index('RESTART\n')
            Startlines[templine + 1] = self.restart_line(step)
        Startfile = open(self.name, 'w')
        Startfile.writelines(Startlines[:-1])
        Startfile.close()

        #Append actions to schedule
        with open(self.name, 'a') as input:
            # Write the action
            input.write('\n')
            input.write('WCONINJE\n')
            input.write(schedule[0])
            input.write('/\n')
            input.write(' \n')
            input.write('WCONPROD\n')
            input.write(schedule[1])
            input.write(schedule[2])
            input.write('/\n')
            input.write(' \n')
            # Write the date
            input.write('DATES\n')
            input.write(self.dates[step+1]+'/\n')
            input.write('/\n')
            input.write('END')

        #Run OPM with this file
        os.system(self.command)

        #Get new state
        state_ = self.get_new_state()

        #Return new state
        return state_

    def get_new_state(self):
        '''Using the RSM file generated by OPM restarting, get the average pressures in the quadrants using 
        the entire pressure grid
        '''

        #Get Reservoir summary values for state
        carbonsats = self.get_carbonsats()
        pressures = self.get_pressures()

        #Stack these state variables into one tensor which is an image with 5 channels
        state_ = torch.stack((pressures, carbonsats))

        return state_
    
    def get_carbonsats(self):
        '''Get carbon saturations
        '''

        table_breaker = "1                                                                                                                                  \n"

        carbonsats = torch.tensor(())

        #Open RSM file
        with open(self.output, 'r') as file:
            SUM = file.readlines()[0:4010]

        indices = [id for id, item in enumerate(SUM) if item == table_breaker]

        #Get carbon sats. See the structure of the RSM file to understand this loop
        for index in indices:
            carbonsat_row = [float(sat) for sat in SUM[index+9].split()]
            carbonsat_row = torch.tensor(carbonsat_row)
            if index == indices[0]:     
                carbonsats = torch.cat((carbonsats, carbonsat_row[2:]),0)
            elif index == indices[-1]:
                carbonsats = torch.cat((carbonsats, carbonsat_row[1].unsqueeze(0)),0)
            else:
                carbonsats = torch.cat((carbonsats, carbonsat_row[1:]),0)

        carbonsats = torch.reshape(carbonsats, (60,60))

        return carbonsats
    
    def get_pressures(self):
        '''Get pressure values
        '''

        table_breaker = "1                                                                                                                                  \n"

        pressures = torch.tensor(())

        #Open RSM file
        with open(self.output, 'r') as file:
            SUM = file.readlines()[4000:8010]

        indices = [id for id, item in enumerate(SUM) if item == table_breaker]

        #Get pressures. See the structure of the RSM file to understand this loop
        for index in indices:
            pressure_row = [float(pressure) for pressure in SUM[index+9].split()]
            pressure_row = torch.tensor(pressure_row)
            if index == indices[0]:
                pressures = torch.cat((pressures, pressure_row[2:]),0)
            elif index == indices[-1]:
                pressures = torch.cat((pressures, pressure_row[1].unsqueeze(0)),0)
            else:
                pressures = torch.cat((pressures, pressure_row[1:]),0)

        pressures = torch.reshape(pressures, (60,60))
        pressures = pressures/self.max_pressure

        return pressures
    
    
   
