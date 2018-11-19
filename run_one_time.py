import gym
import random
import numpy as np
from Agent import Agent
from unity_env import UnityEnv
from collections import OrderedDict
import time

def normalize(vec, high, low):
    if np.any(high == np.inf) or np.any(low == -np.inf):
        return vec
    else:
        return (2*np.array(vec) - np.array(low) - np.array(high))/(np.array(high)-np.array(low))

def denormalize(vec, high, low):
    if np.any(high == np.inf) or np.any(low == -np.inf):
        return vec
    else:
        return (np.array(vec)*(np.array(high)-np.array(low)) + (np.array(high)+np.array(low)))/2

def run_episode_in_test_mode(Agent, Env):
    score = 0
    states = np.array(Env.reset(train_mode = False))  # reset the environment SSS

    actions, actions_perturbed = Agent.choose_action(states)
    actions = actions.detach().numpy()
    actions_perturbed = actions_perturbed.detach().numpy()
    if (actions_perturbed.shape[0] != 1):
        actions_perturbed = actions_perturbed.tolist()
    dones = False * np.ones(len(actions_perturbed))
    t = 0
    while t < 1000:
        next_states, rewards, dones, infos = Env.step(actions_perturbed)
        states = np.array(next_states)

        actions, actions_perturbed = Agent.choose_action(states)
        actions = actions.detach().numpy()
        actions_perturbed = actions_perturbed.detach().numpy()
        if (actions_perturbed.shape[0] != 1):
            actions_perturbed = actions_perturbed.tolist()
        score += np.mean(rewards)  # get the reward
        if np.any(dones) == True:
            break
        # time.sleep(0.005)
        t+= 1
    print(score)


# env_params = {'path' : '/home/pavel/PycharmProjects/Continuous_Control/Reacher_Linux_20/Reacher.x86_64',
#           'worker_id' : 0,
#           'seed' : np.random.randint(1000),
#           'visual_mode' : False,
#           'multiagent_mode' : True}

env_params = {'path' : '/home/pavel/PycharmProjects/Continuous_Control/Reacher_Linux/Reacher.x86_64',
          'worker_id' : 0,
          'seed' : np.random.randint(1000),
          'visual_mode' : False,
          'multiagent_mode' : False}

# env_name = 'Pendulum-v0'
# env = gym.make(env_name) #Pendulum-v0 #MountainCarContinuous-v0 #LunarLanderContinuous-v2

env_name = 'Reacher'
env = UnityEnv(env_params)
observation = env.reset(train_mode = False)
action_space = env.action_space
observation_space = env.observation_space
params = dict()
params['action_dim'] = len(env.action_space.low)
params['state_dim'] =len(observation_space.low)
params['num_episodes'] = 200
params['buffer_size'] = int(1e6)  # replay buffer size
params['batch_size'] = 128          # minibatch size
params['gamma'] = 0.99              # discount factor
params['tau'] = 1e-2                 # for soft update of target parameters
params['eps'] = 0
params['min_eps'] = 0.0
params['eps_decay'] = 1.0
params['lr'] = 0                # learning rate
params['update_every'] = 1          # how often to update the network
params['seed'] = random.randint(0,1000)
params['max_t'] = 10000
params['arch_params_critic'] = OrderedDict(
    {'state_and_action_dims': (params['state_dim'], params['action_dim']),
     'layers': {
         # 'Linear_1': 128, 'ReLU_1': None,
         'Linear_2': 128,  'ReLU_2': None,
         'Linear_3': 64,  'ReLU_3': None,
         # 'Linear_4': 32,'LayerNorm_4': None, 'ReLU_4': None,
         'Linear_5': params['action_dim']
     }
     })

params['arch_params_actor'] = OrderedDict(
        {'state_and_action_dims': (params['state_dim'], params['action_dim']),
         'layers': {
             # 'Linear_1': 128, 'ReLU_1': None,
             'Linear_2': 128, 'ReLU_2': None,
             'Linear_3': 64, 'ReLU_3': None,
             # 'Linear_4': 32, 'LayerNorm_4': None, 'ReLU_4': None,
             'Linear_5': params['action_dim'],  # 'ReLU_5': None,
             'Tanh_1': None
         }
         })

RL_Agent = Agent(params)
RL_Agent.load_weights('Reacher_33.0.prms')
run_episode_in_test_mode(RL_Agent, env)