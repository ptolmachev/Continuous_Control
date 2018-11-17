import numpy as np
import random
from collections import OrderedDict

from QNetwork import QNetwork
from Policy import Policy
from ReplayBuffer import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn as nn


class Agent():
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, gamma, tau, lr, update_every, seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        random.seed(seed)
        self.update_every = update_every
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        arch_params = OrderedDict(
            {'state_and_action_dims': (state_dim, action_dim),
                 'layers' : {
                     'Linear_1': 128, 'ReLU_1': None,
                     'Linear_2': 64, 'ReLU_2': None,
                     'Linear_3': 32, 'ReLU_3': None,
                     'Linear_4': action_dim
                 }
             })
        self.critic_local = QNetwork(seed, arch_params).to(device)  # decision_maker
        self.critic_target = QNetwork(seed, arch_params).to(device) # fixed
        arch_params = OrderedDict(
            {'state_and_action_dims': (state_dim, action_dim),
                 'layers' : {
                     'Linear_1': 128, 'ReLU_1': None,
                     'Linear_2': 64, 'ReLU_2': None,
                     'Linear_3': 32, 'ReLU_3': None,
                     'Linear_4': action_dim, 'Tanh_1' : None
                 }
             })
        self.actor_local = Policy(seed, arch_params).to(device)
        self.actor_target = Policy(seed, arch_params).to(device)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=self.lr)
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.t_step = 0

    def memorize_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1)

    def learn_from_past_experiences(self):
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()  # self.update_every
                self.update_Qnet_and_policy(experiences)

    def choose_action(self,state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state.astype(dtype = np.float)).float().to(device)
        action_to_take = self.actor_local(state)
        return action_to_take
#her
    def update_Qnet_and_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma*Q_targets_next*(1 - dones))  # if done == True: second term is equal to 0
        Q_expected = self.critic_local(states, actions)
        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()
        predicted_actions = self.actor_local(states) # new predicted actions, not the ones stored in buffer
        loss_func = nn.MSELoss()
        loss_critic = loss_func(Q_expected, Q_targets.detach())
        loss_actor = -self.critic_local(states, predicted_actions).mean()
        loss_critic.backward()
        loss_actor.backward()
        self.optimizer_critic.step()
        self.optimizer_actor.step()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def update_eps(self):
        self.actor_local.eps*= self.actor_local.eps_decay
        self.actor_target.eps*= self.actor_target.eps_decay

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
