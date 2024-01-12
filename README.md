# Carbon Storage Project

This project is my part of a broader Reinforcement Learning Project. Note that due to privacy reasons, I could not include all code.

## Broader Problem

Carbon storage (or carbon sequestration) has been proposed as part of a solution to climate change. Carbon is removed from the atmosphere by injecting it in bedrock well below the Earth's surface. This is an optimization problem, as we want to maximize the amount of carbon stored, without increasing the pressure and saturation of Carbon in the bedrock to the point that other problems (such as faults) could arise. Therefore, in the RL setup, the states represent the saturation and pressure across the bedrock. This is represented as a 2x60x60 tensor. The actions represent the Carbon injection rate, and the brine extraction rate. The problem with this RL environment is that it is model-based, and determining the next state based on a state-action pair involves solving some complicated geophysics PDEs that are solved using a computationally slow and expensive third-party software. This makes learning infeasible.

## Transition Surrogate Model

To get around this challenge, a surrogate model which approximates the state transition function is needed. Due to the spatial nature of this task, this involved translating from a 4-channel 60x60 image to a 2-channel 60x60 image. 

First, training data (i.e. (state, action, state) tuples) was needed. The first phase of the project involved connecting to the third-party software and generating many examples. Since the software is slow and many examples were needed, this required moving generation code to Compute Canada (a high-performance compute service for researchers) and parallelizing all the code. This code can be found here (https://github.com/willematack/OPMGenerator).

In phase two, we implemented image-to-image translation models. First, we applied U-Net - a convolutional autoencoder model. We used a architecture similar to that in [1], as shown in the image below:
![image](https://github.com/willematack/CarbonStorage/assets/44038988/12217864-87b2-4b4d-bfbd-fa637a6dbdce)

We have been able to achieve a very low MSE of <1% when translating the states. Here is an example result comparing the real state transition, and the simulated one (still working on improving the pressure grid):
![image](https://github.com/willematack/CarbonStorage/assets/44038988/5c337012-b7d6-4f94-b84d-8a899638b52c)

## Files

The full file structure is not included in this repo due to privacy, but some interesting files include SimulatorCNN.py which shows the U-Net architecture, or Train.py where learning, evaluation and visualization methods are included.

Thanks for checking out this project!!


## References
[1] Sun, A. Y. (2020). Optimal carbon storage reservoir management through deep reinforcement learning. Applied
Energy, 278:115660
