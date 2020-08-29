import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class ReplayMemory(Dataset):

    def __init__(self, max_len, observation_space, agent_space, action_space):
        self.max_len = max_len
        self.observation_space = observation_space
        self.agent_space = agent_space
        self.action_space = action_space

        self.states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.agent_states = torch.zeros([max_len] + list(agent_space), dtype=torch.float32)
        self.actions = torch.zeros([max_len] + list(action_space), dtype=int)
        self.rewards = torch.zeros(max_len, dtype=torch.float32)
        self.new_states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.new_agent_states = torch.zeros([max_len] + list(agent_space), dtype=torch.float32)
        self.dones = torch.zeros(max_len, dtype=bool)

        self.states.requires_grad = False
        self.agent_states.requires_grad = False
        self.actions.requires_grad = False
        self.rewards.requires_grad = False
        self.new_states.requires_grad = False
        self.new_agent_states.requires_grad = False
        self.dones.requires_grad = False
        self.start = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.states[idx], self.agent_states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.new_agent_states[idx], self.dones[idx]

    def random_access(self, n):
        indices = random.sample(range(len(self)), n)
        return self[indices]

    def _extend_unsafe(self, states, agent_states, actions, rewards, new_states, new_agent_states, done, add):

        begin = self.start
        end = begin + add
        self.states[begin:end] = torch.from_numpy(states[:add])
        self.agent_states[begin:end] = torch.from_numpy(agent_states[:add])
        self.actions[begin:end] = torch.from_numpy(actions[:add])
        self.rewards[begin:end] = torch.from_numpy(rewards[:add])
        self.new_states[begin:end] = torch.from_numpy(new_states[:add])
        self.new_agent_states[begin:end] = torch.from_numpy(new_agent_states[:add])
        self.dones[begin:end] = torch.ones(add) * done

    def extend(self, states, agent_states, actions, rewards, new_states, new_agent_states, done):

        add = min(self.max_len - self.start, len(actions[0]))
        if actions[1] is not None:
            actions = np.stack((actions[0], actions[1]), axis=-1)
        else:
            actions = np.stack((actions[0], np.ones(actions[0].shape)), axis=-1)

        self._extend_unsafe(states, agent_states, actions, rewards, new_states, new_agent_states, done, add)
        self.size = max(self.size, min(self.max_len, self.start + add))
        self.start = (self.start + add) % self.max_len
        if add != len(actions):
            self.extend(states[add:], agent_states[add:], actions[add:], rewards[add:], new_states[add:], new_agent_states[add:], done)
