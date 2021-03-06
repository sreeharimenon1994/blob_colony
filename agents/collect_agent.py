from collections import deque
from typing import Optional, Tuple
import random
import numpy as np
from numpy import ndarray
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from agents.agent import Agent
from environment.pheromone import Pheromone
from environment.base import Base
from agents.replay_memory import ReplayMemory

MODEL_NAME = 'Collect_Agent'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 264
UPDATE_TARGET_EVERY = 1

class Model(nn.Module):
    def __init__(self, observation_space, agent_space, mem_size, rotations, pheromones):
        super(Model, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim

        self.agent_input_size = 1
        for dim in agent_space:
            self.agent_input_size *= dim

        self.mem_size = mem_size

        power = 5
        self.layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 2**(2 + power))
        self.layer2 = nn.Linear(2**(2 + power), 2**(3 + power))
        self.layer3 = nn.Linear(2**(3 + power), 2**(1 + power))
        self.layer4 = nn.Linear(2**(1 + power), self.input_size + self.agent_input_size + self.mem_size)

        self.rotation_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 2**(2 + power))
        self.rotation_layer2 = nn.Linear(2**(2 + power), 2**(3 + power))
        self.rotation_layer3 = nn.Linear(2**(3 + power), rotations)

        self.pheromone_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 2**(1 + power))
        self.pheromone_layer2 = nn.Linear(2**(1 + power), pheromones)

        self.memory_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 2**(2 + power))
        self.memory_layer2 = nn.Linear(2**(2 + power), 2**(2 + power))
        self.memory_layer3 = nn.Linear(2**(2 + power), self.mem_size)
        self.forget_layer = nn.Linear(2**(2 + power), self.mem_size)

    def forward(self, state, agent_state):
        old_memory = agent_state[:, self.agent_input_size:]
        all_input = torch.cat([state.view(-1, self.input_size), agent_state.view(-1, self.agent_input_size + self.mem_size)], dim=1)

        general = torch.relu(self.layer1(all_input))
        general = torch.relu(self.layer2(general))
        general = torch.relu(self.layer3(general))
        general = self.layer4(general)

        rotation = self.rotation_layer1(general + all_input)
        rotation = self.rotation_layer2(rotation)
        rotation = self.rotation_layer3(rotation)

        pheromone = self.pheromone_layer1(general + all_input)
        pheromone = self.pheromone_layer2(pheromone)

        memory = self.memory_layer1(general + all_input)
        memory = self.memory_layer2(memory)
        new_memory = torch.tanh(self.memory_layer3(memory))
        forget_fact = torch.sigmoid(self.forget_layer(memory))
        new_memory = new_memory * forget_fact + old_memory * (1 - forget_fact)

        return rotation, pheromone, new_memory


class CollectAgent(Agent):
    def __init__(self, epsilon=0.1, dis=0.5, rotations=3, pheromones=3, lr=1e-4):
        super(CollectAgent, self).__init__("collect_agent")

        self.lr = lr

        self.epsilon = epsilon
        self.dis = dis
        self.rotations = rotations
        self.pheromones = pheromones

        self.model = None
        self.target_model = None
        self.criterion = None
        self.optimizer = None

        # An array with last n steps for training
        self.replay_memory = None

        # Used to count when to update target network with main network's weights
        self.update_target = 0
        self.state = None

        self.mem_size = 20
        self.agent_and_mem_space = None
        self.previous_memory = None

    def setup(self, base: Base, trained_model: Optional[str] = None):
        super(CollectAgent, self).setup(base, trained_model)

        self.previous_memory = torch.zeros((base.blobs.n_blobs, self.mem_size))
        self.agent_and_mem_space = [2 + self.mem_size]

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, self.observation_space, self.agent_and_mem_space, self.action_space)
        self.state = torch.zeros([base.blobs.n_blobs] + list(self.observation_space), dtype=torch.float32)

        # Main model
        self.model = Model(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones)
        self.target_model = Model(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if trained_model is not None:
            self.load_model(trained_model)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def initialize(self, base: Base):
        base.blobs.activate_all_pheromones(
            np.ones((self.n_blobs, len([obj for obj in base.perceived_objects if isinstance(obj, Pheromone)]))) * 10)

    def train(self, itr_done: bool, step: int) -> float:
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 0

        states, agent_state, actions, rewards, new_states, new_agent_state, done = self.replay_memory.random_access(MINIBATCH_SIZE)

        with torch.no_grad():
            rotation_t, pheromones_t, _ = self.target_model(new_states, new_agent_state)
            rotation, pheromones, _ = self.model(states, agent_state)

            rotation_t = torch.max(rotation_t, dim=1).values
            tmp = rewards + self.dis * rotation_t * ~done
            rotation[np.arange(len(rotation)), actions[:, 0].tolist()] = tmp[np.arange(len(rotation))]

            pheromones_t = torch.max(pheromones_t, dim=1).values
            tmp = rewards + self.dis * pheromones_t * ~done
            pheromones[np.arange(len(pheromones)), actions[:, 1].tolist()] = tmp[np.arange(len(pheromones))]

        output = self.model(states, agent_state)
        loss_r = self.criterion(output[0], rotation)
        loss_pher = self.criterion(output[1], pheromones)
        loss = loss_r + loss_pher

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if itr_done:
            self.update_target += 1

        if self.update_target >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.update_target = 0

        return loss.item()

    def update_replay_memory(self, states: ndarray, agent_state: ndarray,
                             actions: Tuple[Optional[ndarray], Optional[ndarray]], rewards: ndarray,
                             new_states: ndarray, new_agent_states: ndarray, done: bool):
        self.replay_memory.extend(states,
                                  np.hstack([agent_state, self.previous_memory]),
                                  (actions[0] + self.rotations // 2, actions[1]),
                                  rewards,
                                  new_states,
                                  np.hstack([new_agent_states, actions[2]]),
                                  done)

    def get_action(self, state: ndarray, agent_state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        if random.random() > self.epsilon or not training:
            with torch.no_grad():
                qs_rotation, qs_pheromones, self.previous_memory = self.target_model(torch.Tensor(state), torch.cat([torch.Tensor(agent_state), self.previous_memory], dim=1))
                action_rot = torch.max(qs_rotation, dim=1).indices.numpy()
                action_phero = torch.max(qs_pheromones, dim=1).indices.numpy()
            rotation = action_rot - self.rotations // 2
            pheromone = action_phero
        else:
            rotation = np.random.randint(low=0, high=self.rotations, size=self.n_blobs) - self.rotations // 2
            pheromone = np.random.randint(low=0, high=self.pheromones, size=self.n_blobs)

        return rotation, pheromone, self.previous_memory.numpy()

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), './agents/models/' + file_name)

    def load_model(self, file_name: str):
        self.model.load_state_dict(torch.load('./agents/models/' + file_name))
        self.target_model.load_state_dict(torch.load('./agents/models/' + file_name))
