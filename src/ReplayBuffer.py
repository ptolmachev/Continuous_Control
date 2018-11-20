import random
import torch
import numpy as np

from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):

        self.__buffer_size = buffer_size
        self.__batch_size = batch_size

        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__memory = deque(maxlen=buffer_size)

    def get_batch_size(self):
        return self.__batch_size

    def is_ready(self):
        return len(self) >= self.__batch_size

    def add(self, state, action, reward, next_state, done):
        self.__memory.append(self.__experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.__memory, k=self.__batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        dones_flatten = dones.view(dones.numel(), -1)
        rewards_flatten = rewards.view(rewards.numel(), -1)
        return states, actions, rewards_flatten, next_states, dones_flatten

    def __len__(self):
        return len(self.__memory)


class Buffer:
    def __init__(self, params):
        batch_size = params['batch_size']
        buffer_size = batch_size
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__memory = deque(maxlen=buffer_size)

    def get_batch_size(self):
        return self.__batch_size

    def is_ready(self):
        return len(self) >= self.__batch_size

    def add(self, state, action, reward, next_state, done):
        self.__memory.append(self.__experience(state, action, reward, next_state, done))

    def get_data(self):

        states = torch.from_numpy(np.vstack([e.state for e in self.__memory if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in self.__memory if e is not None])).float().to(device)
        rewards = np.vstack([e.reward for e in self.__memory if e is not None])
        next_states = torch.from_numpy(np.vstack([e.next_state for e in self.__memory if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in self.__memory if e is not None]).astype(np.uint8)).float().to(device)
        self.__memory.clear()

        dones_flatten = dones.view(dones.numel(), -1)
        # rewards_flatten = rewards.view(rewards.numel(), -1)
        return states, actions, rewards, next_states, dones_flatten

    def __len__(self):
        return len(self.__memory)