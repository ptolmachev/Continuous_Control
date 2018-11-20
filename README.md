# Continuous_Control


### Introduction
This directory contains  the implementation of DDPG (Deep Deterministic Policy Gradient) algorithm appliied for various environments:
*Pendullum-v0*, *MountainCarContinuous-v0* (See information [here](https://github.com/openai/gym/wiki/Leaderboard)), *Reacher* (Unity environment, information may be found [here]https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

The DDPG algorithm was firstly presented in the papaer [Lillicrap et. al](https://arxiv.org/abs/1509.02971).
The pseudocode for this algorithm can be summarised as following:
<p align="center">
<img src="https://github.com/ptolmachev/Continuous_Control/blob/master/img/DDPG_algorithm.jpg"/>
</p>

The idea behind the algorithm:
Given the state of an Agent in the Environment, the Policy network returns an action sampled from the continuous action space slightly perturbed by noise for the exploration purposes. 
The QNetwork then evaluates this action given the state (So the networks accepts concatenated vector state-action and returns a single value).


### Code organisation
The implementation is stored in the folder 'src', which includes:
- `interact_and_train.py`- the main file, to run the training of reinforcment learning agent. It includes hyperparameters and fucntion 'interact_and_train' which creates the instances of an Environmet and an Agent and runs their interaction. This file also includes all the hyperparameters
- `Agent.py` - contains the implementation of an agent. 
- `ReplayBuffer.py` - implementation of internal buffer to sample the experiences from it.
- `QNetwork.py` - an ANN to evaluate q-function.
- `Policy.py` - an ANN to evaluate q-function.
- `plotter.py` - generates the plot of scores acquired during the training.
- `run_one_time.py` - Initialises an agent with specified state dictionary and architecture and run visualisation of the agent's performance.
- `unity_env.py` - wrapper to run Unity Environments using the same code as for OpenAi gym Environments

### Hyperparameters


### Performance of a trained agent



### Suggested further improvements
