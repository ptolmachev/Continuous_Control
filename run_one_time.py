import gym
import random
import numpy as np
from Agent import Agent

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

def run_one_episode_in_test_mode(Agent, Env):
    state_low =  env.observation_space.low
    state_high =  env.observation_space.high
    action_low =  env.action_space.low
    action_high = env.action_space.high
    score = 0
    state_real = Env.reset()  # reset the environment SSS
    state = normalize(state_real, state_low, state_high)
    action = Agent.choose_action(state_real).detach()
    action_real = denormalize(action, action_low, action_high)  # AAA
    done = False
    Agent.actor_local.eps = 0.1
    for t in range(max_t):
        Env.render()
        env_info = Env.step(action_real)
        reward = env_info[1]  # RRR
        next_state_real = env_info[0]  # SSS
        next_state = normalize(next_state_real, state_high, state_low)
        done = env_info[2]
        score += reward  # get the reward
        state = next_state
        action = Agent.choose_action(state).detach()  # AAA
        action_real = denormalize(action, action_high, action_low)  # get new action form the next state
        if done:  # exit loop if episode finished
            break
    print(score)

env = gym.make('Pendulum-v0') #Pendulum-v0 #MountainCarContinuous-v0 #LunarLanderContinuous-v2
observation = env.reset()
action_space = env.action_space
observation_space = env.observation_space
action_dim = len(env.action_space.low)
state_dim =len(observation_space.low)
buffer_size = int(2**17)  # replay buffer size
batch_size = 100          # minibatch size
gamma = 0.99              # discount factor
tau = 1e-2                # for soft update of target parameters
lr = 1e-3                 # learning rate
update_every = 4          # how often to update the network
seed = random.randint(0,1000)
max_t = 2000
RL_Agent = Agent(state_dim, action_dim, buffer_size, batch_size, gamma, tau, lr, update_every, seed)
RL_Agent.actor_local.load('Pendulum-v0_-130.0.plc')
RL_Agent.critic_local.load('Pendulum-v0_-130.0.qnt')

run_one_episode_in_test_mode(RL_Agent, env)