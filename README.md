# Continuous_Control


### Introduction
This directory contains  the implementation of DDPG (Deep Deterministic Policy Gradient) algorithm appliied for various environments:
*Pendullum-v0*, *MountainCarContinuous-v0* (See information [here](https://github.com/openai/gym/wiki/Leaderboard)), *Reacher* (Unity environment, information may be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

The DDPG algorithm was firstly presented in the papaer [Lillicrap et. al](https://arxiv.org/abs/1509.02971).
The pseudocode for this algorithm can be summarised as following:
<p align="center">
<img src="https://github.com/ptolmachev/Continuous_Control/blob/master/img/DDPG_algorithm.jpg"/>
</p>

The idea behind the algorithm:

Given the state of an Agent in the Environment, the Policy network returns an action from the continuous action space slightly perturbed by noise for the exploration purposes. 

The QNetwork then evaluates this action given the state (So the networks accepts concatenated vector state-action and returns a single value).


### Code organization
The implementation is stored in the folder 'src', which includes:
- `interact_and_train.py`- the main file, to run the training of reinforcment learning agent. It includes hyperparameters and fucntion 'interact_and_train' which creates the instances of an Environmet and an Agent and runs their interaction. This file also includes all the hyperparameters
- `Agent.py` - contains the implementation of an Agent. 
- `ReplayBuffer.py` - implementation of internal buffer to sample the experiences.
- `QNetwork.py` - an ANN to evaluate Q-function.
- `Policy.py` - an ANN to chose an action given the state.
- `plotter.py` - generates the plot of scores acquired during the training.
- `run_one_time.py` - initializes an Agent with specified state dictionary and architecture and run visualisation of the Agent's performance.
- `unity_env.py` - wrapper to run Unity Environments using the same code as for OpenAi gym Environments

### Hyperparameters
To solve the Reacher environment the following parameters have been used:
```python
# PARAMETERS
params = dict()
params['action_dim'] = len(env.action_space.low)
params['state_dim'] = len(observation_space.low)
params['num_episodes'] = 200        #number of episodes for agent to interact with the environment
params['buffer_size'] = int(1e6)    # replay buffer size
params['batch_size'] = 128          # minibatch size
params['gamma'] = 0.99              # discount factor
params['tau'] = 1e-2                # for soft update of target parameters
params['eps'] = 0.8                 # exploration factor (modifies noise)
params['min_eps'] = 0.001           # min level of noise
min_e = params['min_eps']
e = params['eps']
N = params['num_episodes']
params['eps_decay'] = np.exp(np.log(min_e/e)/(0.8*N)) #decay of the level of the noise after each episode
params['lr'] = 1e-3                 # learning rate
params['update_every'] = 2          # how often to update the network (every update_every timestep)
params['seed'] = random.randint(0,1000)
params['max_t'] = 1000              # restriction on max number of timesteps per each episodes
params['noise_type'] = 'action'     # noise type; can be 'action' or 'parameter'
params['save_to'] = ('../results/' + env_name) # where to save the results to
params['threshold'] = 38            # the score above which the network parameters are saved


#parameters for the Policy (actor) network
params['arch_params_actor'] = OrderedDict(
        {'state_and_action_dims': (params['state_dim'], params['action_dim']),
         'layers': {
             'Linear_1': 128,   'ReLU_1': None,
             'Linear_2': 64,  'ReLU_2': None,
             'Linear_3': params['action_dim'],
             'Tanh_1': None
         }
         })
#parameters for the QNetwork (critic) network
params['arch_params_critic'] = OrderedDict(
    {'state_and_action_dims': (params['state_dim'], params['action_dim']),
     'layers': {
         'Linear_1': 128, 'ReLU_1': None,
         'Linear_2': 64, 'ReLU_2': None,
         'Linear_3': params['action_dim']
     }
     })
```
### Performance of a trained agent

The scores plot

<p align="center">
<img src="https://github.com/ptolmachev/Continuous_Control/blob/master/img/Scores_Reacher.png"/>
</p>

Gif demonstration of a trained Agent
<p align="center">
<img src="https://github.com/ptolmachev/Continuous_Control/blob/master/img/Reacher_20.gif"/>
</p>


### Ideas to try
- The parameter noise needs some fine tuning. Also it'll be interesting to make the noise adpative to sensitivity of the parameters: the greater the gradient with respect to the specific weight, the lower the level of the noise imposed, and vise versa.
- One may try to make the policy network to return the parameters (mean and variance) of the probability distribution from which the action is sampled. Then we may use the mean of the distribution for the update in QNetwork. 
- Reward Shaping for decreasing the jitter of arm (penalizing the Agent for unnessesary movements to make the Agent act smoother). 
- Of course there is always some space for further hyper-parameter tuning.
- Implementing the PPO (Proximal Policy Optimization, [see original paper](https://arxiv.org/abs/1707.06347)) or TRPO (Trust Region Policy Optimization, [see original paper](https://arxiv.org/abs/1502.05477)) and comparing it to DDPG. It's been suggested, that the PPO-family algorithms work better for continuous control.

