from unity_env import UnityEnv
import gym
import random
import numpy as np
from collections import deque
from Agent import Agent
from collections import OrderedDict
import pickle
from plotter import plotter

def normalize(vec, high, low):
    if (np.any(high == np.inf) or np.any(low == -np.inf) or ( np.all(high) == 1 and (np.all(low) == -1) )):
        return vec
    else:
        return (2*np.array(vec) - np.array(low) - np.array(high))/(np.array(high)-np.array(low))

def denormalize(vec, high, low):
    if ( (np.any(high == np.inf) or np.any(low == -np.inf)) or ( (np.all(high) == 1) and (np.all(low) == -1))):
        return vec
    else:
        return (np.array(vec)*(np.array(high)-np.array(low)) + (np.array(high)+np.array(low)))/2

def interact_and_train(Agent, Env, params):
    state_low =  env.observation_space.low
    state_high =  env.observation_space.high
    action_low =  env.action_space.low
    action_high = env.action_space.high
    num_episodes = params['num_episodes']
    max_t = params['max_t']
    save_to = params['save_to']
    threshold = params['threshold']
    scores = []
    scores_window = deque(maxlen=10)
    best_score = -np.inf
    for e in range(num_episodes):
        score = 0
        states = np.array(Env.reset())  # reset the environment SSS
        states = normalize(states, state_high, state_low)

        actions, actions_perturbed = Agent.choose_action(states)
        actions = denormalize(actions.detach().numpy(), action_high, action_low)
        actions_perturbed = denormalize(actions_perturbed.detach().numpy(), action_high, action_low)
        if (len(actions_perturbed.shape) != 1):
            actions_perturbed = actions_perturbed.tolist()
        dones = False*np.ones(len(actions_perturbed))
        t = 0
        while not (np.any(dones) == True):
            t+=1
            next_states, rewards, dones, infos = Env.step(actions_perturbed)
            next_states = normalize(next_states, state_high, state_low)
            if type(states) == list:
                for i in range(states.shape[0]):
                    Agent.memorize_experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            else:
                Agent.memorize_experience(states, actions, rewards, next_states, dones)
            Agent.learn_from_past_experiences()
            states = np.array(next_states)

            actions, actions_perturbed = Agent.choose_action(states)
            actions = denormalize(actions.detach().numpy(), action_high, action_low)
            actions_perturbed = denormalize(actions_perturbed.detach().numpy(), action_high, action_low)
            if (len(actions_perturbed.shape) != 1):
                actions_perturbed = actions_perturbed.tolist()
            score += np.mean(rewards)  # get the reward
            if (np.any(dones) == True) or (t == max_t):
                break
        Agent.update_eps()
        scores.append(score)
        scores_window.append(score)


        print('\rEpisode {}\tAverage Score: {:.2f}\tCurrent Score : {}'.format(e + 1, np.mean(scores_window), score), end="")
        if (e + 1) % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e + 1, np.mean(scores_window)))
        if (np.mean(scores_window) >= threshold and (np.mean(scores_window) > best_score) ):
            best_score = np.mean(scores_window)
            print('\nEnvironment achieved average score {:.2f} in {:d} episodes!'.format(np.mean(scores_window),(e + 1)))
            file_name = str(save_to)  +'_' + str(np.round(np.mean(scores_window), 0)) + str('.prms')
            Agent.save_weights(str(file_name))
            print("environment saved to ", file_name)
    return scores

# ENVIRONMENT

# 20 agents# Uncomment the lines below to run training in environment with 20 agent
env_params = {'path' : '/home/pavel/PycharmProjects/Continuous_Control/Reacher_Linux_20/Reacher.x86_64',
          'worker_id' : 0,
          'seed' : 1234,
          'visual_mode' : False,
          'multiagent_mode' : True}
env_name = 'Reacher'
env = UnityEnv(env_params)

# One agent# Uncomment the lines below to run training in environment with 1 agent
# env_params = {'path' : '/home/pavel/PycharmProjects/Continuous_Control/Reacher_Linux/Reacher.x86_64',
#           'worker_id' : 0,
#           'seed' : 1234,
#           'visual_mode' : False,
#           'multiagent_mode' : False}
# env_name = 'Reacher'
# env = UnityEnv(env_params)

# OpenAI gym env# uncomment these lines to use gym environment
# env_name = 'Pendulum-v0'
# env = gym.make(env_name) #Pendulum-v0 #MountainCarContinuous-v0 #LunarLanderContinuous-v2

observation = env.reset()
action_space = env.action_space
observation_space = env.observation_space

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

params['arch_params_actor'] = OrderedDict(
        {'state_and_action_dims': (params['state_dim'], params['action_dim']),
         'layers': {
             'Linear_1': 128,   'ReLU_1': None,
             'Linear_2': 64,  'ReLU_2': None,
             'Linear_3': params['action_dim'],
             'Tanh_1': None
         }
         })

params['arch_params_critic'] = OrderedDict(
    {'state_and_action_dims': (params['state_dim'], params['action_dim']),
     'layers': {
         'Linear_1': 128, 'ReLU_1': None,
         'Linear_2': 64, 'ReLU_2': None,
         'Linear_3': params['action_dim']
     }
     })

# AGENT

RL_Agent = Agent(params)

#TRAINING OF AN AGENT
scores = interact_and_train(RL_Agent, env, params)

#SAVING THE RESULTS
pickle.dump(scores, open(params['save_to'] + '.pkl', 'wb+'))

#PLOT THE RESULTS
plotter(scores, threshold = 30)