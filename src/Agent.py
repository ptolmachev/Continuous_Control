import numpy as np
from QNetwork import QNetwork
from Policy import Policy
from ReplayBuffer import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn as nn


class Agent():
    def __init__(self, params):
        self.params = params
        self.__state_dim = params['state_dim']
        self.__action_dim = params['action_dim']
        self.__buffer_size = params['buffer_size']
        self.__batch_size = params['batch_size']
        self.__gamma = params['gamma']
        self.__tau = params['tau']
        self.__lr = params['lr']
        self.__update_every = params['update_every']
        eps = params['eps']
        eps_decay = params['eps_decay']
        min_eps = params['min_eps']
        seed = params['seed']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        critic_params = dict()
        critic_params['seed'] = seed
        critic_params['arch_params'] = params['arch_params_critic']
        self.critic_local = QNetwork(critic_params).to(device)
        self.critic_target = QNetwork(critic_params).to(device)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=self.__lr)

        #Policy
        actor_params = dict()
        actor_params['seed'] = seed
        actor_params['arch_params'] = params['arch_params_actor']
        actor_params['noise_type'] = params['noise_type']
        actor_params['eps'] = eps
        actor_params['eps_decay'] = eps_decay
        actor_params['min_eps'] = min_eps
        actor_params['arch_params'] = params['arch_params_actor']
        self.actor_local = Policy(actor_params).to(device)
        self.actor_target = Policy(actor_params).to(device)
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=self.__lr)

        self.__memory = ReplayBuffer(self.__buffer_size, self.__batch_size)
        self.__t_step = 0

    def memorize_experience(self, state, action, reward, next_state, done):
        self.__memory.add(state, action, reward, next_state, done)
        self.__t_step = (self.__t_step + 1)

    def choose_action(self,state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state.astype(dtype = np.float)).to(device)
        action, action_perturbed = self.actor_local(state)
        return action, action_perturbed

    def learn_from_past_experiences(self):
        if self.__t_step % self.__update_every == 0:
            if len(self.__memory) > self.__batch_size:
                experiences = self.__memory.sample()
                self.update_Qnet_and_policy(experiences)

    def update_Qnet_and_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions, next_actions_perturbed = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.__gamma*Q_targets_next*(1 - dones))  # if done == True: second term is equal to 0
        Q_expected = self.critic_local(states, actions)
        loss_func = nn.MSELoss()
        loss_critic = loss_func(Q_expected, Q_targets.detach())

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        predicted_actions, predicted_actions_perturbed = self.actor_local(states) # new predicted actions, not the ones stored in buffer
        loss_actor = -self.critic_local(states, predicted_actions).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def update_eps(self):
        self.actor_local.eps = max(self.actor_local.eps*self.actor_local.eps_decay, self.actor_local.min_eps)
        self.actor_target.eps = max(self.actor_target.eps*self.actor_target.eps_decay, self.actor_target.min_eps)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)

    def save_weights(self, save_to):
        actor_params = {'actor_params': self.actor_local.policy_params,
                'state_dict': self.actor_local.state_dict()}
        critic_params = {'critic_params': self.critic_local.qnet_params,
                'state_dict': self.critic_local.state_dict()}

        file = dict()
        file['critic_params'] = critic_params
        file['actor_params'] = actor_params
        torch.save(file, open(save_to, 'wb'))


    def load_weights(self, load_from):
        checkpoint = torch.load(load_from)
        qnet_params = checkpoint['critic_params']
        policy_params = checkpoint['actor_params']

        self.actor_local = Policy(policy_params['actor_params'])
        self.actor_local.load_state_dict(checkpoint['actor_params']['state_dict'])

        self.critic_local = QNetwork(qnet_params['critic_params'])
        self.critic_local.load_state_dict(checkpoint['critic_params']['state_dict'])
        return self
