import numpy as np
import torch

##Quick hand-made solution of environment
# import gym
# env = gym.make('MountainCarContinuous-v0')
# observation = env.reset()
# action_space = env.action_space
# observation_space = env.observation_space
# # print(action_space.low, action_space.high)# [-1,1]
# # print(action_space.sample())
# velocity_sign = 1
# score = 0
# done = False
# while not done:
#     env.render()
#     info = env.step([velocity_sign]) #take a random action
#     velocity_sign = 1 if info[0][1]>=0 else -1
#     velocity_sign = 1 if info[0][0] <= -0.7 else velocity_sign
#     score += info[1]
#     done = info[2]
# print("achieved_score = {}".format(score))


from unityagents import UnityEnvironment
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Agent import Agent
from scipy.signal import savgol_filter
from unity_env import UnityEnv

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

def distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = np.sqrt(np.mean(mean_diff))
    return dist

def interact_and_train(Agent, Env, num_episodes, max_t, save_to):
    state_low =  env.observation_space.low
    state_high =  env.observation_space.high
    action_low =  env.action_space.low
    action_high = env.action_space.high
    scores = []
    scores_window = deque(maxlen=100)
    best_score = -np.inf
    for e in range(num_episodes):
        score = 0
        state_real = Env.reset()  # reset the environment SSS
        state = normalize(state_real, state_high, state_low)
        action, action_perturbed = Agent.choose_action(state_real)
        action = action.detach().numpy()
        action_perturbed = action_perturbed.detach().numpy()
        action_real = denormalize(action_perturbed, action_high, action_low)  # AAA
        done = False
        for t in range(max_t):
            env_info = Env.step(action_real)
            reward = env_info[1]  #RRR
            next_state_real = env_info[0] #SSS
            next_state = normalize(next_state_real,state_high, state_low)
            done = env_info[2]
            score += reward  # get the reward
            Agent.memorize_experience(state, action, reward, next_state, done)
            Agent.learn_from_past_experiences()
            state = next_state
            action, action_perturbed = Agent.choose_action(state) #AAA
            action = action.detach().numpy()
            action_perturbed = action_perturbed.detach().numpy()
            # if (distance_metric(action, action_perturbed)) > 0.2:
            #     Agent.actor_local.eps /= 1.01
            # else:
            #     Agent.actor_local.eps *= 1.01

            action_real = denormalize(action_perturbed, action_high, action_low)  # get new action form the next state
            if done:  # exit loop if episode finished
                break
        Agent.update_eps()
        scores.append(score)
        scores_window.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tCurrent Score : {}'.format(e + 1, np.mean(scores_window), score), end="")
        if (e + 1) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e + 1, np.mean(scores_window)))
        if (np.mean(scores_window) >= 200 and (np.mean(scores_window) > best_score) ):
            best_score = np.mean(scores_window)
            print('\nEnvironment achieved average score {:.2f} in {:d} episodes!'.format(np.mean(scores_window),(e + 1)))
            file_name_q = str(save_to)  +'_' + str(np.round(np.mean(scores_window), 0)) + str('.qnt')
            file_name_p = str(save_to)  +'_' + str(np.round(np.mean(scores_window), 0)) + str('.plc')
            Agent.critic_local.save(file_name_q)
            Agent.actor_local.save(file_name_p)
            print("environment saved to ", save_to)
    plt.plot(Agent.actor_local.rand_process.log)
    plt.show()
    return scores


params = {'path' : '/home/pavel/PycharmProjects/Continuous_Control/Reacher_Linux/Reacher.x86_64',
          'worker_id' : 0,
          'seed' : 1234,
          'visual_mode' : False,
          'multiagent_mode' : False}

env_name = 'Pendulum-v0'
env = gym.make(env_name) #Pendulum-v0 #MountainCarContinuous-v0 #LunarLanderContinuous-v2

# env_name = 'Reacher'
# env = UnityEnv(params)

observation = env.reset()
action_space = env.action_space
observation_space = env.observation_space

# examine the state space
action_dim = len(env.action_space.low)
state_dim =len(observation_space.low)
num_episodes = 1000
buffer_size = int(2**17)  # replay buffer size
batch_size = 100          # minibatch size
gamma = 0.99              # discount factor
tau = 1e-2                # for soft update of target parameters
lr = 1e-3                 # learning rate
update_every = 4          # how often to update the network
seed = random.randint(0,1000)
max_t = 200

RL_Agent = Agent(state_dim, action_dim, buffer_size, batch_size, gamma, tau, lr, update_every, seed)
scores = interact_and_train(RL_Agent, env, num_episodes, max_t, save_to = ('./' + env_name))
plt.plot(savgol_filter(scores,51,3))
plt.show()